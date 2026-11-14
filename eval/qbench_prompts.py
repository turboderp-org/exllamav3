import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import copy
import itertools
import json
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

from exllamav3 import Config, Generator, Job, Tokenizer, model_init
from exllamav3.util.progress import ProgressBar
from eval.api_source import add_api_args, api_from_args, assistant_message, response_token_ids

"""
Generate a self-sampled, in-domain test trace for qbench: run a set of multi-turn conversations
through a good-precision quant of the target model with default sampling, and store the exact
token ids of every (context, response) pair until at least --min_tokens response tokens have
been collected.

    python eval/qbench_prompts.py \
        -m <model_dir> [model_init options] \
        -o qbench_prompts.json \
        --min_tokens 20000

The point: evaluating quants on external corpora measures divergence on text the model may
never produce itself (for heavily-aligned reasoning models, raw web text is so far out of
distribution that the noise floor inflates and KLD ordering degrades). The model's own sampled
output is in-distribution by definition; a qbench project can point `test_trace` at the output
file and KLD/ppl are then computed only over the sampled (response) token positions, with each
(context, response) pair as one variable-length row.

Reasoning traces are kept in the scored response. For context in follow-up turns the previous
response is cleaned the way deployment would (harmony final channel extracted / <think> blocks
dropped) before re-templating; the stored ids are exactly what the model saw and sampled either
way.

With --tool_frac > 0 (default 0.3) a portion of the conversations define tools (passed through
the chat template's `tools` support) and ask questions expected to elicit a reasoning trace 
plus a tool invocation; scripted tool-role results feed follow-up turns when the model actually
produced a tool call.

--domain and --lang each take one name or a comma-separated list (or "all"), and a trace then
holds every listed domain in every listed language it exists in:

    python eval/qbench_prompts.py -m <model_dir> -o qbench_prompts.json \
        --domain general,literature,tools --lang en,zh,fr --min_tokens 150000

The sets take turns, so a token budget that runs out early still covers all of them, and each
row records the domain and language it came from.

Without hardware for a good quant, --api_model sources the responses from an OpenAI-compatible
chat API instead, OpenRouter by default:

    python eval/qbench_prompts.py -m <model_dir> --api_model deepseek/deepseek-v4.1-flash -o qbench_prompts.json

-m then only provides the tokenizer and chat template: prompts are templated and tokenized
locally, and each finished turn, tool calls included, is rendered back through the template to
recover the token stream the model would have produced (eval/api_source.py). Responses are
requested concurrently, one wave per turn, so the trace can overshoot --min_tokens by up to
--api_concurrency responses.
"""

# Seeds chosen to elicit long reasoning traces plus a substantial final answer, across domains.
# Follow-ups deepen the same conversation (self-review prompts produce natural reasoning).
CONVERSATIONS = [
    ("Four hikers reach a rope bridge at night with one flashlight. They cross in 1, 2, 5 and "
     "10 minutes respectively; the bridge holds two people, and any crossing pair moves at the "
     "slower hiker's pace. The flashlight must be walked back, not thrown. Find the minimum "
     "total crossing time and prove it can't be beaten.",
     ["Re-derive the answer if the times are 1, 3, 4 and 6 minutes, and explain which strategy "
      "changes and why.",
      "Now five hikers: 1, 2, 5, 10 and 12 minutes. What's the optimal time?"]),
    ("Write a Python class implementing a token-bucket rate limiter: configurable capacity and "
     "refill rate, a try_acquire(n) method, monotonic time, no busy-waiting. Explain the design.",
     ["Which edge cases could break it (clock behavior, bursts, fractional refill), and how does "
      "your implementation handle each?",
      "Port it to TypeScript with the same semantics."]),
    ("Estimate how many fuel stations operate in Germany, showing your chain of reasoning from "
     "population and driving habits up to the final number.",
     ["Which single assumption dominates the error budget, and what range does the estimate "
      "cover if that assumption is off by a factor of 2 either way?",
      "Repeat the estimate for Norway, where EV adoption is far higher."]),
    ("Explain why you hear thunder after seeing lightning, first for a curious 10-year-old, "
     "then for an engineering student — the second version should cover the actual acoustics "
     "of the shockwave and why distant thunder rumbles instead of cracking.",
     ["The student asks: why can you sometimes see lightning but never hear any thunder at all?",
      "Write a five-question quiz covering both explanations, with an answer key."]),
    ("Plan a 12-day self-drive trip through New Zealand's South Island for two travelers who "
     "love mountains and wildlife but want to avoid tour-bus crowds, in early March. Day-by-day "
     "itinerary with driving times.",
     ["Rework the plan for a total budget of NZ$4500 excluding flights.",
      "A storm closes the West Coast road on day 6. Replan the remaining days on the fly."]),
    ("Prove that log base 2 of 3 is irrational, then explain exactly which property of the "
     "integers the proof leans on.",
     ["Generalize: for which pairs of integers a, b > 1 is log base a of b rational? Prove your "
      "characterization.",
      "Explain the proof to someone comfortable with fractions but who has never seen a proof "
      "by contradiction."]),
    ("Design a database schema for a multi-location physiotherapy clinic: practitioners, "
     "patients, appointment types, rooms, recurring availability, bookings, cancellations and "
     "waitlists. Give the DDL and justify the key modeling decisions.",
     ["The clinic adds video consultations that don't occupy rooms but still bill. What changes?",
      "Write the four queries the reception desk runs constantly, with indexes to support them."]),
    ("Three roommates share an apartment. Rent is $2,340, split in proportion to bedroom size: "
     "Dana's room is 14 m², Eli's is 11 m², Fern's is 9 m². Utilities add $187 split evenly, "
     "and Eli gets a $40 credit for handling maintenance. Compute what each person owes, to the "
     "cent, and verify the totals reconcile.",
     ["Present the full calculation as a table and double-check every rounding step.",
      "Now redo the split if the utility bill is instead divided in the same proportion as rent."]),
    ("Summarize the main economic arguments for and against a substantially higher minimum "
     "wage, citing the standard competitive and monopsony models, then give your overall read "
     "of the empirical literature.",
     ["Steelman the position your overall read went against, as persuasively as you can.",
      "How do the arguments change for a regional minimum wage indexed to local median income?"]),
    ("Write a 400-word short story about a museum night guard who notices that one painting is "
     "slightly different every morning, with an ending that recontextualizes the whole story.",
     ["Critique your story's use of foreshadowing and rewrite the weakest paragraph.",
      "Retell the same events as a sequence of terse incident reports filed by the guard."]),
]

# Chinese counterparts of the same conversations (same tasks, so a comparison between traces
# isolates the language). Selected with --lang zh; tool specs and scripted tool results stay as they
# are (English JSON), only the user turns change.
CONVERSATIONS_ZH = [
    ("四名徒步者在夜里到达一座绳桥，只有一只手电筒。他们过桥分别需要1、2、5和10分钟；桥一次最多承载两人，"
     "同行的两人必须按较慢者的速度前进。手电筒必须有人拿回来，不能扔过去。求最短的总过桥时间，并证明无法更短。",
     ["如果四人的时间改为1、3、4和6分钟，重新推导答案，并说明策略在哪里变了、为什么。",
      "现在有五名徒步者：1、2、5、10和12分钟。最优时间是多少？"]),
    ("用Python写一个令牌桶限流器类：容量和填充速率可配置，提供try_acquire(n)方法，使用单调时钟，不要忙等待。"
     "解释你的设计。",
     ["哪些边界情况可能让它出错（时钟行为、突发流量、非整数填充），你的实现分别是怎么处理的？",
      "把它移植到TypeScript，保持相同的语义。"]),
    ("估算德国有多少座加油站在营业，从人口和驾驶习惯出发，一步步展示推理链，直到给出最终数字。",
     ["哪一个假设对误差影响最大？如果这个假设偏差两倍（无论偏大还是偏小），估算值的范围是多少？",
      "对挪威重复这个估算，那里的电动车普及率高得多。"]),
    ("解释为什么先看到闪电、后听到雷声：先讲给一个好奇的10岁孩子听，再讲给一个工科大学生听——第二个版本要涵盖"
     "冲击波的实际声学原理，以及为什么远处的雷声是隆隆声而不是炸响。",
     ["这个学生接着问：为什么有时候能看到闪电，却完全听不到雷声？",
      "围绕这两种解释出一份五道题的小测验，并附答案。"]),
    ("为两位热爱山地和野生动物、但想避开旅游大巴人群的旅行者，规划一次12天的新西兰南岛自驾游，时间在三月初。"
     "给出逐日行程和驾驶时间。",
     ["把计划改成总预算4500新西兰元（不含机票）。",
      "第6天一场风暴封闭了西海岸公路。临时重新规划剩下的行程。"]),
    ("证明以2为底3的对数是无理数，然后准确说明这个证明依赖整数的哪一条性质。",
     ["推广：对于哪些整数对a, b > 1，以a为底b的对数是有理数？证明你的刻画。",
      "向一个熟悉分数、但从未见过反证法的人解释这个证明。"]),
    ("为一家多门店物理治疗诊所设计数据库模式：治疗师、患者、预约类型、诊室、周期性的可用时段、预约、取消和候补名单。"
     "给出DDL并说明关键建模决策的理由。",
     ["诊所新增了视频问诊，不占用诊室但仍然计费。需要改什么？",
      "写出前台最常执行的四条查询，并给出支持它们的索引。"]),
    ("三个室友合租一套公寓。房租2340美元，按卧室面积比例分摊：Dana的房间14平方米，Eli的11平方米，Fern的9平方米。"
     "水电费187美元平均分摊，Eli因为负责维修获得40美元抵扣。计算每人应付多少（精确到美分），并核对总额是否对得上。",
     ["把完整计算过程列成表格，并逐一复核每个四舍五入步骤。",
      "如果水电费改为按与房租相同的比例分摊，重新计算。"]),
    ("总结大幅提高最低工资的主要经济学论据（支持与反对），引用标准的竞争模型和买方垄断模型，然后给出你对实证研究"
     "文献的总体判断。",
     ["尽可能有说服力地为你总体判断所不支持的那一方做最强辩护。",
      "如果是与当地收入中位数挂钩的地区性最低工资，这些论据会有什么变化？"]),
    ("写一篇400字的短篇小说：一位博物馆夜班保安发现某幅画每天早上都有细微的不同，结尾要让整个故事的意义发生反转。",
     ["评价你这篇故事中伏笔的运用，并重写最弱的一段。",
      "把同样的事件改写成保安提交的一系列简短事件报告。"]),
]

# French counterparts, on the same terms as the Chinese ones. Selected with --lang fr
CONVERSATIONS_FR = [
    ("Quatre randonneurs arrivent de nuit devant un pont de corde avec une seule lampe torche. Ils "
     "mettent respectivement 1, 2, 5 et 10 minutes à traverser ; le pont supporte deux personnes à "
     "la fois, et deux randonneurs qui traversent ensemble avancent au rythme du plus lent. La "
     "lampe doit être rapportée à pied, pas lancée. Trouve le temps total de traversée minimal et "
     "démontre qu'on ne peut pas faire mieux.",
     ["Refais le raisonnement si les temps sont de 1, 3, 4 et 6 minutes, et explique quelle "
      "stratégie change et pourquoi.",
      "Maintenant cinq randonneurs : 1, 2, 5, 10 et 12 minutes. Quel est le temps optimal ?"]),
    ("Écris une classe Python qui implémente un limiteur de débit à seau de jetons (token bucket) : "
     "capacité et taux de remplissage configurables, une méthode try_acquire(n), une horloge "
     "monotone, pas d'attente active. Explique la conception.",
     ["Quels cas limites pourraient le mettre en défaut (comportement de l'horloge, rafales, "
      "remplissage fractionnaire), et comment ton implémentation gère-t-elle chacun d'eux ?",
      "Porte-le en TypeScript avec la même sémantique."]),
    ("Estime combien de stations-service sont en activité en Allemagne, en détaillant ton "
     "raisonnement depuis la population et les habitudes de conduite jusqu'au chiffre final.",
     ["Quelle hypothèse pèse le plus dans la marge d'erreur, et quelle fourchette couvre "
      "l'estimation si cette hypothèse est fausse d'un facteur 2 dans un sens ou dans l'autre ?",
      "Refais l'estimation pour la Norvège, où les véhicules électriques sont bien plus répandus."]),
    ("Explique pourquoi on entend le tonnerre après avoir vu l'éclair, d'abord pour un enfant "
     "curieux de 10 ans, puis pour un étudiant en école d'ingénieurs. La seconde version doit "
     "couvrir l'acoustique réelle de l'onde de choc et expliquer pourquoi un tonnerre lointain "
     "gronde au lieu de claquer.",
     ["L'étudiant demande : pourquoi voit-on parfois des éclairs sans jamais entendre le moindre "
      "tonnerre ?",
      "Rédige un quiz de cinq questions couvrant les deux explications, avec un corrigé."]),
    ("Planifie un voyage de 12 jours en voiture sur l'île du Sud de la Nouvelle-Zélande pour deux "
     "voyageurs qui adorent la montagne et la faune sauvage mais veulent éviter les foules des "
     "autocars de tourisme, début mars. Itinéraire jour par jour avec les temps de trajet.",
     ["Revois le plan pour un budget total de 4 500 NZ$ hors vols.",
      "Une tempête ferme la route de la côte ouest le sixième jour. Replanifie les jours restants "
      "au pied levé."]),
    ("Démontre que le logarithme en base 2 de 3 est irrationnel, puis explique précisément sur "
     "quelle propriété des entiers repose la démonstration.",
     ["Généralise : pour quels couples d'entiers a, b > 1 le logarithme en base a de b est-il "
      "rationnel ? Démontre ta caractérisation.",
      "Explique la démonstration à quelqu'un qui est à l'aise avec les fractions mais n'a jamais "
      "vu de raisonnement par l'absurde."]),
    ("Concevez un schéma de base de données pour un cabinet de kinésithérapie réparti sur "
     "plusieurs sites : praticiens, patients, types de rendez-vous, salles, disponibilités "
     "récurrentes, réservations, annulations et listes d'attente. Donnez le DDL et justifiez les "
     "principaux choix de modélisation.",
     ["Le cabinet ajoute des consultations vidéo qui n'occupent pas de salle mais sont tout de "
      "même facturées. Qu'est-ce qui change ?",
      "Écrivez les quatre requêtes que l'accueil exécute en permanence, avec les index qui les "
      "accompagnent."]),
    ("Trois colocataires partagent un appartement. Le loyer est de 2 340 $, réparti au prorata de "
     "la surface des chambres : celle de Dana fait 14 m², celle d'Eli 11 m², celle de Fern 9 m². "
     "Les charges ajoutent 187 $ répartis à parts égales, et Eli bénéficie d'un crédit de 40 $ "
     "parce qu'il s'occupe de l'entretien. Calcule ce que doit chaque personne, au centime près, "
     "et vérifie que les totaux concordent.",
     ["Présente le calcul complet sous forme de tableau et revérifie chaque étape d'arrondi.",
      "Refais maintenant la répartition si les charges sont elles aussi divisées dans la même "
      "proportion que le loyer."]),
    ("Résumez les principaux arguments économiques pour et contre un salaire minimum nettement "
     "plus élevé, en vous appuyant sur les modèles classiques de concurrence et de monopsone, puis "
     "donnez votre lecture d'ensemble de la littérature empirique.",
     ["Défendez, de la manière la plus convaincante possible, la position que votre lecture "
      "d'ensemble écartait.",
      "Comment ces arguments changent-ils pour un salaire minimum régional indexé sur le revenu "
      "médian local ?"]),
    ("Écris une nouvelle de 400 mots sur un gardien de nuit de musée qui remarque qu'un tableau "
     "est légèrement différent chaque matin, avec une fin qui éclaire toute l'histoire sous un "
     "jour nouveau.",
     ["Fais la critique de l'usage des indices annonciateurs dans ta nouvelle et réécris le "
      "paragraphe le plus faible.",
      "Raconte les mêmes événements sous la forme d'une série de rapports d'incident laconiques "
      "rédigés par le gardien."]),
]

# Tool-calling conversations: (tools, seed prompt, follow-ups). Tools are OpenAI-style specs
# passed through the chat template. Follow-ups tagged "tool" are scripted tool results injected
# as tool-role messages — used only if the previous response actually contained a tool call —
# while "user" follow-ups continue the conversation normally.
WEATHER_TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Get the current weather and short-term forecast for a location",
        "parameters": {"type": "object", "properties": {
            "location": {"type": "string", "description": "City name, e.g. 'Bergen, Norway'"},
            "days": {"type": "integer", "description": "Forecast days, 1-7"}},
            "required": ["location"]}}},
    {"type": "function", "function": {
        "name": "get_air_quality",
        "description": "Get the current air quality index for a location",
        "parameters": {"type": "object", "properties": {
            "location": {"type": "string"}},
            "required": ["location"]}}},
]

FINANCE_TOOLS = [
    {"type": "function", "function": {
        "name": "get_stock_quote",
        "description": "Get the latest price and daily change for a stock ticker",
        "parameters": {"type": "object", "properties": {
            "ticker": {"type": "string", "description": "Stock symbol, e.g. 'ASML'"}},
            "required": ["ticker"]}}},
    {"type": "function", "function": {
        "name": "convert_currency",
        "description": "Convert an amount between currencies at the current exchange rate",
        "parameters": {"type": "object", "properties": {
            "amount": {"type": "number"},
            "from_currency": {"type": "string"},
            "to_currency": {"type": "string"}},
            "required": ["amount", "from_currency", "to_currency"]}}},
]

CALENDAR_TOOLS = [
    {"type": "function", "function": {
        "name": "list_events",
        "description": "List calendar events in a date range",
        "parameters": {"type": "object", "properties": {
            "start_date": {"type": "string", "description": "ISO date"},
            "end_date": {"type": "string", "description": "ISO date"}},
            "required": ["start_date", "end_date"]}}},
    {"type": "function", "function": {
        "name": "create_event",
        "description": "Create a calendar event",
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string"},
            "start": {"type": "string", "description": "ISO datetime"},
            "duration_minutes": {"type": "integer"},
            "attendees": {"type": "array", "items": {"type": "string"}}},
            "required": ["title", "start", "duration_minutes"]}}},
]

SEARCH_TOOLS = [
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the web and return a list of results with titles and snippets",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "num_results": {"type": "integer"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "fetch_page",
        "description": "Fetch the readable text content of a web page",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string"}},
            "required": ["url"]}}},
]

SQL_TOOLS = [
    {"type": "function", "function": {
        "name": "run_query",
        "description": "Run a read-only SQL query against the analytics warehouse. Tables: "
                       "orders(id, customer_id, created_at, total_usd, status), "
                       "customers(id, name, country, signup_date), "
                       "refunds(id, order_id, amount_usd, created_at)",
        "parameters": {"type": "object", "properties": {
            "sql": {"type": "string"}},
            "required": ["sql"]}}},
]

TOOL_CONVERSATIONS = [
    (WEATHER_TOOLS,
     "I'm flying to Bergen on Thursday for a long weekend of hiking. Should I pack for rain, "
     "and is the air quality okay for someone with mild asthma? Check before answering.",
     [("tool", json.dumps({"location": "Bergen, Norway", "current": {"temp_c": 9, "conditions":
        "light rain"}, "forecast": [{"day": 1, "high_c": 11, "low_c": 6, "precip_mm": 14},
        {"day": 2, "high_c": 10, "low_c": 5, "precip_mm": 22}, {"day": 3, "high_c": 12,
        "low_c": 7, "precip_mm": 3}]})),
      ("tool", json.dumps({"location": "Bergen, Norway", "aqi": 18, "category": "good",
        "pm25": 4.1})),
      ("user", "Given all that, plan which of the three days is best for the longest hike.")]),
    (FINANCE_TOOLS,
     "I hold 120 shares of ASML and 300 shares of NVO, and I think in euros. What is my "
     "position worth right now in EUR? Look up what you need.",
     [("tool", json.dumps({"ticker": "ASML", "price_usd": 812.40, "change_pct": -1.2})),
      ("tool", json.dumps({"ticker": "NVO", "price_usd": 94.75, "change_pct": 0.8})),
      ("tool", json.dumps({"amount": 125913.0, "from_currency": "USD", "to_currency": "EUR",
        "result": 116218.7, "rate": 0.923})),
      ("user", "If ASML drops another 5% and the euro strengthens 2%, what's the new value?")]),
    (CALENDAR_TOOLS,
     "Find a free 90-minute slot for a design review with Priya and Tomás next Tuesday or "
     "Wednesday between 9:00 and 17:00, and book it. My calendar is the source of truth — "
     "check it first.",
     [("tool", json.dumps({"events": [
        {"title": "Standup", "start": "2026-08-25T09:00", "duration_minutes": 30},
        {"title": "1:1 with manager", "start": "2026-08-25T11:00", "duration_minutes": 60},
        {"title": "Sprint planning", "start": "2026-08-26T13:00", "duration_minutes": 120}]})),
      ("tool", json.dumps({"created": True, "event_id": "evt_8842", "title": "Design review",
        "start": "2026-08-25T14:00", "duration_minutes": 90})),
      ("user", "Actually Priya is out Tuesday afternoon. Move it to the best Wednesday slot.")]),
    (SEARCH_TOOLS,
     "What is the current status of the ITER fusion project's first-plasma timeline, and how "
     "many times has it slipped? Search for recent information rather than answering from memory.",
     [("tool", json.dumps({"results": [
        {"title": "ITER updates baseline schedule", "snippet": "The ITER Organization confirmed "
         "a revised research plan, with initial operations now targeted for 2034 and full "
         "deuterium-tritium operation in 2039...", "url": "https://example.org/iter-baseline"},
        {"title": "A history of ITER delays", "snippet": "Originally slated for first plasma in "
         "2016 under the 2001 design, the project moved to 2020, then 2025, before the latest "
         "revision...", "url": "https://example.org/iter-history"}]})),
      ("user", "Summarize the main engineering causes behind the most recent slip.")]),
    (SQL_TOOLS,
     "Using the warehouse, work out our net revenue (orders minus refunds) by country for the "
     "last full quarter, and flag any country where refunds exceed 10% of gross. Write and run "
     "the query, then interpret the results.",
     [("tool", json.dumps({"columns": ["country", "gross_usd", "refunds_usd", "net_usd"],
        "rows": [["US", 412000, 18400, 393600], ["DE", 148000, 21500, 126500],
                 ["JP", 96500, 2100, 94400], ["BR", 40200, 6900, 33300]]})),
      ("user", "DE and BR look bad. Write a follow-up query that breaks their refunds down by "
       "order status so we can see where the problem is.")]),
    (WEATHER_TOOLS,
     "Compare this weekend's conditions in Chamonix and Zermatt and tell me which is the "
     "better choice for high-altitude photography. Gather the data you need first.",
     [("tool", json.dumps({"location": "Chamonix, France", "forecast": [
        {"day": 1, "high_c": 4, "cloud_pct": 85, "wind_kmh": 45},
        {"day": 2, "high_c": 6, "cloud_pct": 30, "wind_kmh": 20}]})),
      ("tool", json.dumps({"location": "Zermatt, Switzerland", "forecast": [
        {"day": 1, "high_c": 3, "cloud_pct": 20, "wind_kmh": 15},
        {"day": 2, "high_c": 5, "cloud_pct": 10, "wind_kmh": 10}]}))]),
]

# Chinese user turns for the tool conversations; tool specs and scripted results are shared
TOOL_SEEDS_ZH = [
    "我周四要飞往卑尔根，去徒步度一个长周末。我需要带雨具吗？对于有轻度哮喘的人来说，那里的空气质量还可以吗？回答前先查一下。",
    "我持有120股ASML和300股NVO，我习惯用欧元来计算。我的持仓现在值多少欧元？需要什么数据自己去查。",
    "在下周二或周三9:00到17:00之间，为我、Priya和Tomás找一个90分钟的空档开设计评审会，并预订下来。以我的日历为准——先查日历。",
    "ITER聚变项目“首次等离子体”时间表的最新状态如何？它一共推迟过几次？请搜索最新信息，不要凭记忆回答。",
    "用数据仓库算出上一个完整季度按国家划分的净收入（订单减退款），并标出退款超过总收入10%的国家。写出并运行查询，然后解读结果。",
    "比较这个周末霞慕尼和采尔马特的天气状况，告诉我哪个更适合高海拔摄影。先收集你需要的数据。",
]
TOOL_USER_FOLLOWUPS_ZH = [
    "综合这些信息，规划一下三天里哪一天最适合走最长的那条徒步路线。",
    "如果ASML再跌5%，欧元升值2%，新的市值是多少？",
    "其实Priya周二下午不在。把会议挪到周三最合适的时段。",
    "总结最近一次推迟背后的主要工程原因。",
    "DE和BR看起来不妙。写一个后续查询，按订单状态拆分这两个国家的退款，看看问题出在哪里。",
    None,
]

# French user turns for the tool conversations
TOOL_SEEDS_FR = [
    "Je prends l'avion pour Bergen jeudi pour un long week-end de randonnée. Dois-je prévoir des "
    "affaires de pluie, et la qualité de l'air convient-elle à quelqu'un qui a un asthme léger ? "
    "Vérifie avant de répondre.",
    "Je détiens 120 actions ASML et 300 actions NVO, et je raisonne en euros. Combien vaut ma "
    "position en ce moment, en EUR ? Va chercher les données dont tu as besoin.",
    "Trouve un créneau libre de 90 minutes pour une revue de conception avec Priya et Tomás mardi "
    "ou mercredi prochain, entre 9 h et 17 h, et réserve-le. Mon agenda fait foi : consulte-le "
    "d'abord.",
    "Où en est le calendrier du premier plasma du projet de fusion ITER, et combien de fois "
    "a-t-il été repoussé ? Fais une recherche pour trouver des informations récentes plutôt que "
    "de répondre de mémoire.",
    "À partir de l'entrepôt de données, calcule notre chiffre d'affaires net (commandes moins "
    "remboursements) par pays sur le dernier trimestre complet, et signale tout pays où les "
    "remboursements dépassent 10 % du brut. Écris et exécute la requête, puis interprète les "
    "résultats.",
    "Compare les conditions de ce week-end à Chamonix et à Zermatt et dis-moi laquelle des deux "
    "est le meilleur choix pour de la photographie en haute altitude. Rassemble d'abord les "
    "données nécessaires.",
]
TOOL_USER_FOLLOWUPS_FR = [
    "Compte tenu de tout cela, détermine lequel des trois jours convient le mieux à la plus "
    "longue randonnée.",
    "Si ASML perd encore 5 % et que l'euro s'apprécie de 2 %, quelle est la nouvelle valeur ?",
    "En fait, Priya est absente mardi après-midi. Déplace la réunion sur le meilleur créneau de "
    "mercredi.",
    "Résume les principales causes techniques du dernier report en date.",
    "DE et BR font mauvaise figure. Écris une requête complémentaire qui ventile leurs "
    "remboursements par statut de commande, pour qu'on voie d'où vient le problème.",
    None,
]

# --domain literature: close reading, criticism and writing in verse and prose (no tools)
CONVERSATIONS_LIT = [
    ("Give a close reading of Shelley's \"Ozymandias\": how the nested narrators (the speaker, the "
     "traveller, the sculptor, the king) frame its irony, and what the sonnet form contributes.",
     ["Compare it with Horace Smith's sonnet on the same subject, written in the same contest. Why "
      "is Shelley's the one that lasted?",
      "Write your own Petrarchan sonnet on a ruined modern monument, then annotate its rhyme scheme "
      "and the volta."]),
    ("Emily Dickinson's dashes and slant rhymes are often called unconventional. Using \"Because I "
     "could not stop for Death\" and \"I heard a Fly buzz – when I died\", explain what they do "
     "that conventional punctuation and full rhyme could not.",
     ["How did her early editors 'correct' these poems, and what was lost in the process?",
      "Write three short poems in her manner about modern objects: a phone, a parking lot, a "
      "smoke alarm."]),
    ("Explain the difference between free verse, blank verse and accentual verse, with a short "
     "example passage of each that you write yourself, scanned to show the stresses.",
     ["Now scan the opening four lines of Milton's Paradise Lost and point out every place where "
      "the iambic pentameter is varied.",
      "Rewrite your free-verse example as a villanelle, keeping its central image."]),
    ("Discuss the narrative voice in Moby-Dick: how reliable is Ishmael, why does he sometimes "
     "narrate scenes he cannot have witnessed, and what do the cetology chapters do for the novel?",
     ["Contrast Ahab's speeches with Ishmael's narration. Where does Melville borrow from "
      "Shakespeare, and to what effect?",
      "Write a 300-word scene in Melville's style in which Starbuck confronts Ahab one last time."]),
    ("Trace the development of the haiku from the hokku of renga through Bashō, Buson and Issa to "
     "Shiki's reforms, and explain the roles of the kigo and the kireji.",
     ["Why is the 5-7-5 syllable count a poor fit for English, and what do English-language haiku "
      "poets do instead?",
      "Compose a sequence of six haiku following one day in late autumn, with a note on each "
      "seasonal word."]),
    ("Compare the endings of Austen's Persuasion and Pride and Prejudice: how does each novel "
     "bring its lovers to an understanding, and what does the difference say about Austen's "
     "development as a writer?",
     ["Austen famously uses free indirect discourse. Pick a passage you remember and show how "
      "it works line by line.",
      "Write a letter from Anne Elliot to her sister Mary, in period style, about Captain "
      "Wentworth's return."]),
    ("What makes Borges's \"The Garden of Forking Paths\" both a spy story and a philosophical "
     "essay? Walk through how the labyrinth metaphor operates on each level.",
     ["Relate the story to his other labyrinths: \"The Library of Babel\" and \"The House of "
      "Asterion\".",
      "Write a 250-word Borgesian fable about an infinite archive of unsent letters."]),
    ("Analyze T. S. Eliot's \"The Love Song of J. Alfred Prufrock\": the epigraph from Dante, the "
     "fragmented imagery, the refrains, and why the poem is considered a turning point for "
     "modernist verse.",
     ["How does Eliot's \"objective correlative\" from his Hamlet essay help explain Prufrock's "
      "technique?",
      "Write a dramatic monologue in which Prufrock, now old, revisits the same streets."]),
    ("Compare three English translations of the opening of Homer's Odyssey (Chapman, Pope, and "
     "a modern one such as Fagles or Wilson) from memory as best you can, and discuss what each "
     "translator prioritizes.",
     ["Emily Wilson rendered polytropos as 'complicated'. Defend and criticize that choice.",
      "Translate the same opening yourself into unrhymed English hexameters and explain your "
      "compromises."]),
    ("Discuss the role of the unreliable narrator in Kazuo Ishiguro's The Remains of the Day: how "
     "does Stevens reveal what he refuses to say, and how does the novel's structure as a road "
     "trip support that?",
     ["Compare Stevens with the narrator of Ford Madox Ford's The Good Soldier.",
      "Write the diary entry Miss Kenton might have written the evening after their final meeting "
      "at the bus stop."]),
]


def longctx_conversations():
    """--domain longctx: long documents (fiction, technical documentation, source code) as the
    first user turn, followed by questions that need the whole document. Excerpts are cut at
    paragraph boundaries to about 30k characters"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    def excerpt(paths, start = 0, chars = 30000):
        text = ""
        for path in paths.split("+"):
            with open(os.path.join(root, path), "r", encoding = "utf-8") as f:
                text += f.read() + "\n\n"
        a = text.find("\n\n", start) + 2 if start else 0
        b = text.rfind("\n\n", a, a + chars)
        return text[a:b if b > a else a + chars].strip()

    fiction = [
        "Summarize this excerpt scene by scene, naming every character who appears in each scene.",
        "Trace how the relationship between the two most important characters changes over the "
        "excerpt, quoting the passages where it turns.",
        "Which details early in the excerpt pay off later? List them with both the setup and the "
        "payoff.",
    ]
    technical = [
        "Summarize this document section by section, and explain who it is written for.",
        "List every configuration option, flag or environment variable mentioned, with what it "
        "does in one line each.",
        "Where is the document unclear, inconsistent or likely out of date? Point to the exact "
        "passages and propose corrected wording.",
    ]
    code = [
        "Explain what this code does, module by module and function by function, and how the "
        "pieces fit together.",
        "Find the three places most likely to hide a bug or an edge case the code doesn't handle, "
        "and explain the failure in each.",
        "Write docstrings for the five most important functions that currently lack a complete "
        "one, matching the existing style.",
    ]
    docs = [
        ("an excerpt from a work of fiction", "eval/eval_texts/pride_prejudice_mod.txt", 0, fiction),
        ("an excerpt from a work of fiction", "eval/eval_texts/pride_prejudice_mod.txt", 330000, fiction),
        ("an excerpt from a work of fiction", "eval/eval_texts/variable_man_mod.txt", 0, fiction),
        ("an excerpt from a work of fiction", "eval/eval_texts/variable_man_mod.txt", 70000, fiction),
        ("an excerpt from a work of fiction", "eval/eval_texts/illustrious_client.txt", 0, fiction),
        ("a technical document", "doc/env_vars.md", 0, technical),
        ("a technical document", "README.md+doc/convert.md", 0, technical),
        ("a source file", "exllamav3/tokenizer/tokenizer.py", 0, code),
    ]
    convs = []
    for kind, path, start, questions in docs:
        doc = excerpt(path, start)
        seed = f"Here is {kind}:\n\n```\n{doc}\n```\n\n{questions[0]}"
        convs.append((seed, questions[1:]))
    return convs


DOMAINS = ["general", "literature", "longctx", "tools"]
LANGS = ["en", "zh", "fr"]

GENERAL_CONVERSATIONS = {"en": CONVERSATIONS, "zh": CONVERSATIONS_ZH, "fr": CONVERSATIONS_FR}
TOOL_USER_TURNS = {"zh": (TOOL_SEEDS_ZH, TOOL_USER_FOLLOWUPS_ZH), "fr": (TOOL_SEEDS_FR, TOOL_USER_FOLLOWUPS_FR)}

# The languages each domain exists in. The agentic sessions of the tools domain are English only
DOMAIN_LANGS = {"general": LANGS, "literature": ["en"], "longctx": ["en"], "tools": LANGS}


def tool_conversations(lang: str):
    """The scripted tool conversations with their user turns in the given language"""
    if lang == "en":
        return TOOL_CONVERSATIONS
    seeds, followups = TOOL_USER_TURNS[lang]
    convs = []
    for (tools, _, fus), seed, fu in zip(TOOL_CONVERSATIONS, seeds, followups):
        fus = [(kind, fu if kind == "user" else content) for kind, content in fus]
        assert all(c is not None for _, c in fus)
        convs.append((tools, seed, fus))
    return convs


def conversations_for(domain: str, lang: str, tool_frac: float):
    """One domain's conversations in one language, ready to sample. Only the general domain mixes the two kinds:
    about tool_frac of its conversations define tools, interleaved evenly so a small token budget that stops early
    still samples both"""
    def plain(convs):
        return [{"tools": None, "messages": [], "turns": [("user", seed)] + [("user", fu) for fu in fus]}
                for seed, fus in convs]
    def tooled(convs):
        return [{"tools": tools, "messages": [], "turns": [("user", seed)] + list(fus)}
                for tools, seed, fus in convs]

    if domain == "literature":
        return plain(CONVERSATIONS_LIT)
    if domain == "longctx":
        return plain(longctx_conversations())
    if domain == "tools":
        return tooled(tool_conversations(lang))
    assert domain == "general", domain
    base = plain(GENERAL_CONVERSATIONS[lang])
    if tool_frac <= 0:
        return base
    if tool_frac >= 1.0:
        return tooled(tool_conversations(lang))
    tool_convs = tool_conversations(lang)
    n_tools = min(len(tool_convs), max(1, round(tool_frac / (1 - tool_frac) * len(base))))
    tool_convs = tooled(tool_convs[:n_tools])
    convs = []
    acc = 0.0
    ratio = len(tool_convs) / len(base)
    for b in base:
        convs.append(b)
        acc += ratio
        while acc >= 1.0 and tool_convs:
            convs.append(tool_convs.pop(0))
            acc -= 1.0
    return convs + tool_convs


def build_conversations(args, tokenizer, template_vars: dict):
    """The conversations to sample: one set per listed domain in each listed language it exists in. The sets take
    turns in the result, so a token budget that stops early still covers all of them"""
    # The tools domain holds every scripted tool conversation, so next to it the general set keeps none
    tool_frac = 0 if "tools" in args.domain else args.tool_frac
    sets = []
    for domain in args.domain:
        for lang in args.lang:
            if lang not in DOMAIN_LANGS[domain]:
                continue
            convs = conversations_for(domain, lang, tool_frac)
            if domain == "tools" and lang == "en":
                convs += agentic_conversations(tokenizer, template_vars, args.max_context)
            for conv in convs:
                conv["domain"], conv["lang"] = domain, lang
            sets.append(convs)
            print(f" -- {domain}/{lang}: {len(convs)} conversations")
    convs = [c for group in itertools.zip_longest(*sets) for c in group if c is not None]
    # Independent resamples of the whole set (each repeat continues its own conversations)
    return [copy.deepcopy(c) for _ in range(args.repeats) for c in convs]


# --domain tools: besides the scripted tool conversations, real agentic sessions (the spec_decode
# prompts) cut before each assistant turn whose templated context fits --max_context; the model
# samples its own next action there
AGENTIC_TRANSCRIPTS = ["eval/prompts/agentic_code_29.json", "eval/prompts/agentic_curl_16.json"]

def agentic_conversations(tokenizer, template_vars: dict, max_context: int):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    convs = []
    for path in AGENTIC_TRANSCRIPTS:
        with open(os.path.join(root, path), "r") as f:
            data = json.load(f)
        messages = data["messages"]
        for msg in messages:
            for tc in msg.get("tool_calls") or []:
                func = tc.get("function", {})
                if isinstance(func.get("arguments"), str):
                    func["arguments"] = json.loads(func["arguments"])
        for i, msg in enumerate(messages):
            if msg["role"] != "assistant" or i < 2:
                continue
            ids = tokenizer.hf_chat_template(messages[:i], add_generation_prompt = True,
                                             tools = data.get("tools"), **template_vars)
            if ids.shape[-1] > max_context:
                break
            last = messages[i - 1]
            convs.append({"tools": data.get("tools"), "messages": list(messages[:i - 1]),
                          "turns": [(last["role"], last["content"])], "prefilled": True})
    return convs


# Response substrings that indicate the model actually invoked a tool (family-specific markup,
# checked on the special-token decode). If none match, scripted tool results are not injected
TOOL_CALL_MARKERS = ["<tool_call>", "<|tool_call", "tool▁call", "<function", "to=functions",
                     "<|python_tag|>", "[TOOL_CALLS]"]

DEFAULT_TEMPLATE_VARS = dict(enable_thinking = True, reasoning_effort = "high")

col_default = "[0m"
col_yellow = "[33;1m"
col_blue = "[34;1m"


def clean_response_for_context(tokenizer, response_ids: torch.Tensor) -> str:
    """Previous-turn assistant content for re-templating: reasoning segments dropped the way a
    deployed chat loop would drop them"""
    full = tokenizer.decode(response_ids, decode_special_tokens = True)
    if "<|channel|>final<|message|>" in full:              # gpt-oss harmony
        text = full.rsplit("<|channel|>final<|message|>", 1)[1]
        for stop in ("<|return|>", "<|end|>", "<|call|>"):
            text = text.split(stop)[0]
        return text.strip()
    plain = tokenizer.decode(response_ids, decode_special_tokens = False)
    if "</think>" in plain:                                # qwen/deepseek style
        plain = plain.rsplit("</think>", 1)[1]
    return plain.strip()


def resolve_template_vars(tokenizer, template_vars):
    """Some templates reject specific variables (e.g. qwen3.8 only accepts reasoning_effort
    xhigh/medium/low and errors on 'high'). Drop offending keys, falling back to the template's
    own defaults"""
    probe = [{"role": "user", "content": "hi"}]
    def works(vars_):
        try:
            tokenizer.hf_chat_template(probe, add_generation_prompt = True, **vars_)
            return True
        except Exception:
            return False
    template_vars = dict(template_vars)
    while template_vars and not works(template_vars):
        for k in list(template_vars):
            if works({kk: v for kk, v in template_vars.items() if kk != k}):
                print(f" !! Chat template rejects {k} = {template_vars[k]!r}; using template default")
                del template_vars[k]
                break
        else:
            print(" !! Chat template rejects all variable combinations; using template defaults")
            template_vars = {}
    return template_vars


def conv_template_kwargs(conv: dict, template_vars: dict) -> dict:
    return template_vars if conv["tools"] is None else dict(template_vars, tools = conv["tools"])


def sample_local(generator, config, tokenizer, args, convs, template_vars):
    rows = []
    total_in = total_out = 0
    # Turn-major order: every conversation's first turn before any follow-ups, so a small token
    # budget still spreads across all domains instead of exhausting on the first conversations
    max_turns = max(len(c["turns"]) for c in convs)
    schedule = [(c_idx, t_idx) for t_idx in range(max_turns)
                for c_idx in range(len(convs)) if t_idx < len(convs[c_idx]["turns"])]
    with ProgressBar("Sampling", args.min_tokens) as pb:
        for c_idx, t_idx in schedule:
            if total_out >= args.min_tokens:
                break
            conv = convs[c_idx]
            if t_idx > 0 and len(conv["messages"]) < 2 * t_idx:
                continue   # earlier turn of this conversation failed or was skipped
            messages = conv["messages"]
            kind, content = conv["turns"][t_idx]
            if kind == "tool":
                # Scripted tool result: only meaningful if the previous response actually
                # contained a tool invocation
                prev = messages[-1]["content"] if messages else ""
                prefilled = conv.get("prefilled") and t_idx == 0
                if not prefilled and not any(m in prev for m in TOOL_CALL_MARKERS):
                    continue
                messages.append({"role": "tool", "content": content})
            else:
                messages.append({"role": "user", "content": content})
            template_kwargs = conv_template_kwargs(conv, template_vars)
            try:
                input_ids = tokenizer.hf_chat_template(
                    messages, add_generation_prompt = True, **template_kwargs)
            except Exception as e:
                print(f" !! Templating failed for conversation {c_idx} turn {t_idx}: {e}")
                messages.pop()
                continue
            job = Job(
                input_ids = input_ids,
                max_new_tokens = args.max_new_tokens,
                stop_conditions = config.eos_token_id_list,
                decode_special_tokens = True,
            )
            generator.enqueue(job)
            chunks = []
            total_temp = total_out
            text_temp = ""
            while generator.num_remaining_jobs():
                for result in generator.iterate():
                    if result["stage"] == "streaming" and "token_ids" in result:
                        chunks.append(result["token_ids"])
                        total_temp += result["token_ids"].shape[-1]
                        text = result["text"]
                        text_temp += text
                        if "\n" in text:
                            print(text_temp, end = "")
                            text_temp = ""
                            pb.update(min(total_temp, args.min_tokens))
            # Flush the tail after the last newline -- for tool-calling turns this is
            # exactly where the <tool_call> markup sits
            if text_temp:
                print(text_temp, end = "")
            response_ids = torch.cat(chunks, dim = -1)[0] if chunks else torch.empty(0, dtype = torch.long)
            if response_ids.numel() == 0:
                messages.pop()
                continue
            rows.append({
                "conversation": c_idx,
                "turn": t_idx,
                "domain": conv["domain"],
                "lang": conv["lang"],
                "tools": conv["tools"] is not None,
                "input_ids": input_ids[0].tolist(),
                "response_ids": response_ids.tolist(),
            })
            total_in += input_ids.shape[-1]
            total_out += response_ids.numel()
            pb.update(min(total_out, args.min_tokens))
            # Keep tool-call markup in context: cleaned only of reasoning segments, so a
            # scripted tool-role turn can follow it the way a deployed loop's transcript would
            messages.append({
                "role": "assistant",
                "content": clean_response_for_context(tokenizer, response_ids),
            })
            print(f"\n{col_blue}---------------------------------------------------------------{col_default}\n")
            print(f"Input tokens:  {col_yellow}{total_in:6,}{col_default}")
            print(f"Output tokens: {col_yellow}{total_out:6,}{col_default}")
            print(f"\n{col_blue}---------------------------------------------------------------{col_default}\n")
    return rows, total_in, total_out


def wire_messages(messages: list) -> list:
    """A conversation that starts from a transcript, as the API takes it: the templates want tool call arguments
    as dicts, the API as JSON strings"""
    wire = copy.deepcopy(messages)
    for msg in wire:
        for tc in msg.get("tool_calls") or []:
            tc["function"]["arguments"] = json.dumps(tc["function"]["arguments"])
    return wire


def open_tool_calls(messages: list) -> list:
    """The calls of the last response that have no result yet"""
    answered = 0
    for msg in reversed(messages):
        if msg["role"] != "tool":
            return (msg.get("tool_calls") or [])[answered:]
        answered += 1
    return []


def next_api_turn(conv: dict) -> bool:
    """Adds what comes next in the script to the conversation, or returns False if that can't follow the last
    response. In a local conversation a tool call is just markup in the context. An API refuses further messages
    until each call of the last response has its result, by id, so a response that makes more calls than the script
    has results for takes as many as there are and an error for the rest."""
    messages, wire, turns, t = conv["messages"], conv["wire"], conv["turns"], conv["next_turn"]
    calls = open_tool_calls(messages)
    is_tool = turns[t][0] == "tool"
    if is_tool and not calls:
        return False
    for call in calls:
        content = json.dumps({"error": "no result available"})
        if t < len(turns) and turns[t][0] == "tool":
            content = turns[t][1]
            t += 1
        messages.append({"role": "tool", "tool_call_id": call["id"], "name": call["function"]["name"], "content": content})
        wire.append({"role": "tool", "tool_call_id": call["id"], "content": content})
    if not is_tool:
        messages.append({"role": "user", "content": turns[t][1]})
        wire.append(messages[-1])
        t += 1
    conv["next_turn"] = t
    return True


def sample_api(api, args, tokenizer, convs, template_vars):
    """Turn-major like the local path, a wave of concurrent requests at a time. Each conversation keeps two
    histories: "messages" is what the template renders, tool calls structured and reasoning dropped the way the
    local path drops it, and "wire" is what the API gets, with its tool call turns sent back as it returned them."""
    rows = []
    total_in = total_out = 0
    for conv in convs:
        conv["wire"] = wire_messages(conv["messages"])
        conv["next_turn"] = 0
    active = list(range(len(convs)))

    def request(c_idx, seed):
        # Requests already running still finish, but what's left of the wave isn't billed
        if total_out >= args.min_tokens:
            return None
        return api.complete(convs[c_idx]["wire"], seed, convs[c_idx]["tools"])

    with ProgressBar("Sampling", args.min_tokens) as pb:
        while active and total_out < args.min_tokens:
            pending = {}
            for c_idx in active:
                conv = convs[c_idx]
                if conv["next_turn"] >= len(conv["turns"]) or not next_api_turn(conv):
                    continue
                template_kwargs = conv_template_kwargs(conv, template_vars)
                try:
                    input_ids = tokenizer.hf_chat_template(conv["messages"], add_generation_prompt = True, **template_kwargs)
                except Exception as e:
                    print(f" !! Templating failed for conversation {c_idx} turn {conv['next_turn'] - 1}: {e}")
                    continue
                pending[c_idx] = (input_ids, template_kwargs)
            active = []
            print(f" -- {col_blue}{len(pending)}{col_default} requests")

            wave_rows = []
            with ThreadPoolExecutor(max_workers = args.api_concurrency) as pool:
                futures = {}
                for c_idx in pending:
                    seed = zlib.crc32(f"{c_idx}|{convs[c_idx]['next_turn']}".encode())
                    futures[pool.submit(request, c_idx, seed)] = c_idx
                for fut in as_completed(futures):
                    c_idx = futures[fut]
                    conv = convs[c_idx]
                    t_idx = conv["next_turn"] - 1
                    input_ids, template_kwargs = pending[c_idx]
                    try:
                        r = fut.result()
                        if r is None:
                            continue
                        if not (r["content"] or r["reasoning"] or r["tool_calls"]):
                            raise ValueError("empty response")
                        assistant = assistant_message(r["content"], r["reasoning"], r["tool_calls"])
                        ids = response_token_ids(tokenizer, conv["messages"], assistant, template_kwargs)
                    except Exception as e:
                        print(f"    conv {c_idx:3} turn {t_idx}: {col_yellow}failed{col_default} ({str(e)[:120]})")
                        continue
                    wave_rows.append({
                        "conversation": c_idx,
                        "turn": t_idx,
                        "domain": conv["domain"],
                        "lang": conv["lang"],
                        "tools": conv["tools"] is not None,
                        "input_ids": input_ids[0].tolist(),
                        "response_ids": ids,
                    })
                    total_in += input_ids.shape[-1]
                    total_out += len(ids)
                    pb.update(min(total_out, args.min_tokens))
                    print(f"    conv {c_idx:3} turn {t_idx}: {col_yellow}{len(ids):5}{col_default} tokens "
                          f"({r['finish_reason']}, reasoning {len(r['reasoning'] or '')} chars)")
                    conv["messages"].append({k: v for k, v in assistant.items() if k in ("role", "content", "tool_calls")})
                    conv["wire"].append(r["message"] if r["tool_calls"] else {"role": "assistant", "content": r["content"]})
                    active.append(c_idx)
            # Completion order would make the trace depend on request timing
            rows += sorted(wave_rows, key = lambda row: row["conversation"])
            active.sort()

    if total_out < args.min_tokens:
        print(f" !! Collected only {total_out:,} of {args.min_tokens:,} response tokens; "
              f"consider a higher --repeats or --max_new_tokens")
    return rows, total_in, total_out


@torch.inference_mode()
def main(args):
    api = None
    if args.api_model:
        # Responses from the API; the local model is only its tokenizer and chat template
        config = Config.from_directory(args.model_dir)
        tokenizer = Tokenizer.from_config(config)
        generator = None
        api = api_from_args(args)
    else:
        model, config, cache, tokenizer, draft_model, draft_config, draft_cache = model_init.init(args)
        generator = Generator(
            model = model,
            cache = cache,
            tokenizer = tokenizer,
            draft_model = draft_model,
            draft_cache = draft_cache,
            num_draft_tokens = args.num_draft_tokens,
            sparse_draft_head = getattr(args, 'sparse_draft_head', 0),
            ngram_match_min = args.ngram_match_min,
            ngram_corpus = args.ngram_corpus,
            dynamic_draft_tokens = args.dynamic_draft,
            draft_confidence = args.draft_confidence,
            max_chunk_size = args.chunk_size,
        )

    TEMPLATE_VARS = DEFAULT_TEMPLATE_VARS.copy()
    TEMPLATE_VARS.update(args.template_vars)
    TEMPLATE_VARS = resolve_template_vars(tokenizer, TEMPLATE_VARS)
    if api is not None:
        api.request_reasoning(TEMPLATE_VARS)

    convs = build_conversations(args, tokenizer, TEMPLATE_VARS)

    if api is not None:
        rows, total_in, total_out = sample_api(api, args, tokenizer, convs, TEMPLATE_VARS)
        api.print_usage()
    else:
        rows, total_in, total_out = sample_local(generator, config, tokenizer, args, convs, TEMPLATE_VARS)

    out = {
        "model": args.model_dir,
        "vocab_size": tokenizer.actual_vocab_size,
        "template_vars": TEMPLATE_VARS,
        "tool_frac": args.tool_frac,
        "domain": ",".join(args.domain),
        "lang": ",".join(args.lang),
        "meta": {"rows": len(rows), "input_tokens": total_in, "output_tokens": total_out},
        "rows": rows,
    }
    if api is not None:
        out["api_model"] = args.api_model
    with open(args.output, "w") as f:
        json.dump(out, f)
    print(f" -- {len(rows)} rows, {total_in:,} input + {total_out:,} output tokens -> {args.output}")


def name_list(choices: list):
    """argparse type: one name or a comma-separated list out of choices, or "all" for every one"""
    def parse(s):
        names = choices if s == "all" else [n.strip() for n in s.split(",")]
        if not all(n in choices for n in names):
            raise argparse.ArgumentTypeError(f"expected names out of {', '.join(choices)}, comma-separated, or all")
        return list(dict.fromkeys(names))
    return parse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev = False)
    model_init.add_args(parser, default_cache_size = 32768, add_draft_model_args = True)
    parser.add_argument("-o", "--output", type = str, required = True, help = "Output JSON file")
    parser.add_argument("--min_tokens", type = int, default = 20000, help = "Stop once this many response tokens are collected")
    parser.add_argument("--max_new_tokens", type = int, default = 4096, help = "Per-turn generation cap")
    parser.add_argument("--tool_frac", type = float, default = 0.3, help = "Approximate fraction of conversations that define tools and elicit tool calls, default: 0.3; 0 disables")
    parser.add_argument("-tv", "--template_vars", type = json.loads, default = {}, help = 'JSON dict of chat template variables, merged over the defaults')
    parser.add_argument("--domain", type = name_list(DOMAINS), default = ["general"],
                        help = "Conversation sets, one or a comma-separated list, or all: general (default; mixed reasoning tasks plus --tool_frac tool "
                               "calls), literature (poetry and literary criticism, no tools), longctx (questions over long documents, no tools), tools "
                               "(scripted tool conversations plus agentic-session cut points up to --max_context). With tools in the list, the general "
                               "set leaves the tool conversations to it")
    parser.add_argument("--max_context", type = int, default = 12288, help = "--domain tools: longest templated context of an agentic cut point, default: 12288")
    parser.add_argument("--repeats", type = int, default = 1, help = "Sample the whole conversation set this many times (independent samples per repeat), default: 1")
    parser.add_argument("--lang", type = name_list(LANGS), default = ["en"],
                        help = "Languages of the user turns, one or a comma-separated list, or all: en (default), zh, fr. The same conversations in each; "
                               "tool specs/results stay English. The general and tools domains exist in every language (agentic sessions in English "
                               "only), literature and longctx in English only, so a list that includes en covers every domain")
    add_api_args(parser)
    args = parser.parse_args()
    for domain in args.domain:
        if not set(args.lang) & set(DOMAIN_LANGS[domain]):
            parser.error(f"--domain {domain} exists only in {', '.join(DOMAIN_LANGS[domain])}, which --lang must then include")
    main(args)
