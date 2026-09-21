import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json

import torch

from exllamav3 import Generator, Job, model_init
from exllamav3.util.progress import ProgressBar

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

# Chinese counterparts of the same conversations (same tasks, so a comparison between the two traces
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


def conversations_for_lang(lang: str):
    """(base conversations, tool conversations) for --lang"""
    if lang == "en":
        return CONVERSATIONS, TOOL_CONVERSATIONS
    assert lang == "zh", lang
    tool_convs = []
    for (tools, seed, fus), seed_zh, fu_zh in zip(TOOL_CONVERSATIONS, TOOL_SEEDS_ZH, TOOL_USER_FOLLOWUPS_ZH):
        fus_zh = [(kind, fu_zh if kind == "user" else content) for kind, content in fus]
        assert all(c is not None for _, c in fus_zh)
        tool_convs.append((tools, seed_zh, fus_zh))
    return CONVERSATIONS_ZH, tool_convs

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


@torch.inference_mode()
def main(args):
    model, config, cache, tokenizer, draft_model, draft_config, draft_cache = model_init.init(args)
    generator = Generator(
        model = model,
        cache = cache,
        tokenizer = tokenizer,
        draft_model = draft_model,
        draft_cache = draft_cache,
        num_draft_tokens = args.num_draft_tokens,
        ngram_match_min = args.ngram_match_min,
        dynamic_draft_tokens = args.dynamic_draft,
        draft_confidence = args.draft_confidence,
        max_chunk_size = args.chunk_size,
    )

    TEMPLATE_VARS = DEFAULT_TEMPLATE_VARS.copy()
    TEMPLATE_VARS.update(args.template_vars)
    TEMPLATE_VARS = resolve_template_vars(tokenizer, TEMPLATE_VARS)

    # Base conversations plus an optional tool-calling portion (~args.tool_frac of the total),
    # interleaved evenly so a small token budget that stops early still samples both kinds
    CONVS, TOOL_CONVS = conversations_for_lang(args.lang)
    base = [{"tools": None, "messages": [],
             "turns": [("user", seed)] + [("user", fu) for fu in fus]}
            for seed, fus in CONVS]
    convs = []
    if args.tool_frac >= 1.0:
        # Tools only
        convs = [{"tools": tools, "messages": [],
                  "turns": [("user", seed)] + list(fus)}
                 for tools, seed, fus in TOOL_CONVS]
    elif args.tool_frac > 0:
        n_tools = min(len(TOOL_CONVS),
                      max(1, round(args.tool_frac / (1 - args.tool_frac) * len(CONVS))))
        tool_convs = [{"tools": tools, "messages": [],
                       "turns": [("user", seed)] + list(fus)}
                      for tools, seed, fus in TOOL_CONVS[:n_tools]]
        acc = 0.0
        ratio = len(tool_convs) / len(base)
        for b in base:
            convs.append(b)
            acc += ratio
            while acc >= 1.0 and tool_convs:
                convs.append(tool_convs.pop(0))
                acc -= 1.0
        convs += tool_convs
    else:
        convs = base

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
                if not any(m in prev for m in TOOL_CALL_MARKERS):
                    continue
                messages.append({"role": "tool", "content": content})
            else:
                messages.append({"role": "user", "content": content})
            template_kwargs = dict(TEMPLATE_VARS)
            if conv["tools"] is not None:
                template_kwargs["tools"] = conv["tools"]
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

    out = {
        "model": args.model_dir,
        "vocab_size": tokenizer.actual_vocab_size,
        "template_vars": TEMPLATE_VARS,
        "tool_frac": args.tool_frac,
        "lang": args.lang,
        "meta": {"rows": len(rows), "input_tokens": total_in, "output_tokens": total_out},
        "rows": rows,
    }
    with open(args.output, "w") as f:
        json.dump(out, f)
    print(f" -- {len(rows)} rows, {total_in:,} input + {total_out:,} output tokens -> {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev = False)
    model_init.add_args(parser, default_cache_size = 32768, add_draft_model_args = True)
    parser.add_argument("-o", "--output", type = str, required = True, help = "Output JSON file")
    parser.add_argument("--min_tokens", type = int, default = 20000, help = "Stop once this many response tokens are collected")
    parser.add_argument("--max_new_tokens", type = int, default = 4096, help = "Per-turn generation cap")
    parser.add_argument("--tool_frac", type = float, default = 0.3, help = "Approximate fraction of conversations that define tools and elicit tool calls, default: 0.3; 0 disables")
    parser.add_argument("-tv", "--template_vars", type = json.loads, default = {}, help = 'JSON dict of chat template variables, merged over the defaults')
    parser.add_argument("--lang", type = str, default = "en", choices = ["en", "zh"], help = "Language of the user turns: en (default) or zh (same conversations in Chinese; tool specs/results stay English)")
    main(parser.parse_args())
