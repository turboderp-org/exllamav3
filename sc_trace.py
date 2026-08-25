import sys, os

import argparse
import json
import random
import zlib

import torch
from safetensors.torch import save_file

from exllamav3 import Generator, Job, model_init
from exllamav3.util.progress import ProgressBar
from eval.qbench_prompts import clean_response_for_context, DEFAULT_TEMPLATE_VARS

"""
Generate a large self-sampled, in-domain trace, sized for use as quantization calibration data:
~250 rows x 2048 tokens by default, spanning a wide range of subjects, task formats and 
languages.

    python sc_trace.py \
        -m <model_dir> [model_init options] \
        -o trace.json \
        -co trace.safetensors

The trace should be produced from the unquantized model or a high-bitrate quant (e.g. 6 bpw,
minimally distorted), then reused for measurement (measure.py), recipe optimization and
conversion of all lower-bitrate quants of the same model.

If the seed set alone doesn't reach the token target, additional epochs re-run the same
conversations with different per-job sampling seeds.
"""

# (seed prompt, [follow-ups]). Follow-ups deepen the same conversation; most conversations are
# 1-2 turns so the token budget spreads across subjects instead of re-prefilling long contexts.
# Non-English prompts are written in the target language to elicit responses in that language.
CONVERSATIONS = [

    # --- Reasoning / puzzles (from qbench_prompts) ---
    ("A farmer needs to cross a river with a wolf, a goat, a cabbage, and a second goat that "
     "eats wolves. The boat holds the farmer and two items. Find a safe crossing sequence.",
     ["Double-check your solution step by step and fix any mistakes."]),
    ("Estimate how many piano tuners work in Chicago, showing your reasoning.",
     ["Which of your assumptions is most fragile, and how would the estimate change if it's off by 2x?"]),
    ("Three friends split a restaurant bill of $127.50. Alice pays twice what Bob pays, and Carol "
     "pays $7.50 more than Bob. During payment, a 15% service charge is added to the total. How "
     "much does each person actually pay?",
     ["Verify the arithmetic carefully and present the answer as a table."]),
    ("Five houses in a row are painted different colors, and each owner has a different pet and "
     "drink. Given: the red house is left of the green one; the dog owner drinks tea; the cat "
     "lives in the first house; the milk drinker lives in the middle house; the parrot owner "
     "lives next to the cat. Invent two more consistent clues, then solve for a full assignment.",
     ["Is your solution unique? If not, show a second valid assignment."]),
    ("You have 12 balls, one of which is heavier or lighter. Using a balance scale three times, "
     "determine which ball is odd and whether it's heavy or light. Explain the full decision tree.",
     []),
    ("A snail climbs 3 meters up a wall each day and slides down 2 meters each night. The wall is "
     "10 meters tall. On which day does it reach the top? Now generalize to climb a, slide b, height h.",
     ["Write a one-line closed-form expression and test it on five random cases."]),

    # --- Mathematics ---
    ("Prove that the square root of 2 is irrational, then explain where the same proof breaks "
     "down if you try it on the square root of 4.",
     ["Generalize: for which integers n is sqrt(n) irrational? Prove it."]),
    ("Evaluate the integral of x^2 * e^(-x) from 0 to infinity, showing each integration by parts.",
     ["Now derive the general formula for the integral of x^n * e^(-x) and relate it to the gamma function."]),
    ("A fair coin is flipped until two heads appear in a row. What is the expected number of flips? "
     "Set up the recurrence and solve it.",
     ["Extend to three heads in a row, and give the general pattern for n heads."]),
    ("Explain the difference between pointwise and uniform convergence of function sequences, with "
     "a concrete example of a sequence that converges pointwise but not uniformly.",
     []),
    ("Find all integer solutions to 3x + 7y = 1, then describe the full solution set of any linear "
     "Diophantine equation ax + by = c.",
     ["When does ax + by = c have no solutions? Give three concrete examples."]),
    ("A tetrahedron has vertices at (0,0,0), (1,0,0), (0,1,0), (0,0,1). Compute its volume, "
     "surface area, and the radius of its inscribed sphere.",
     []),
    ("Explain Bayes' theorem using a medical test example: a disease affects 1 in 1000 people, "
     "the test has 99% sensitivity and 95% specificity. What does a positive result mean?",
     ["Now compute the effect of a second, independent positive test."]),
    ("Show that the harmonic series diverges, using two different proofs.",
     ["How fast does the partial sum grow? Derive the connection to the natural logarithm."]),

    # --- Coding ---
    ("Write a Python function that finds the longest palindromic substring in O(n) time, with "
     "an explanation of the algorithm.",
     ["What are the edge cases and how does your implementation handle them?"]),
    ("Design a database schema for a public library system: books, copies, members, loans, "
     "holds, fines. Give the DDL and explain the key decisions.",
     ["Write the five queries the front desk will run most often."]),
    ("Implement an LRU cache in Python with O(1) get and put, first using OrderedDict, then from "
     "scratch with a doubly-linked list and dict.",
     ["Make the from-scratch version thread-safe and explain the locking strategy."]),
    ("Write a Rust function that parses a simplified INI file format into a nested HashMap, "
     "handling comments, whitespace, and duplicate keys sensibly. Include unit tests.",
     []),
    ("Explain what this regex does and rewrite it to be readable with named groups and comments: "
     r"^(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)$",
     ["Now write a regex that validates an IPv6 address, and explain why this is much harder."]),
    ("Here is a JavaScript function with a bug:\n\n"
     "```js\nasync function fetchAll(urls) {\n  const results = [];\n  urls.forEach(async (u) => {\n"
     "    const r = await fetch(u);\n    results.push(await r.json());\n  });\n  return results;\n}\n```\n\n"
     "Explain why it returns an empty array and fix it two different ways.",
     ["What changes if the endpoints are rate-limited to 5 concurrent requests?"]),
    ("Write a SQL query against tables orders(id, customer_id, created_at, total) and "
     "customers(id, name, country) that returns each country's top 3 customers by lifetime "
     "spend, including ties. Explain the window function used.",
     ["Rewrite it for SQLite 3.24, which lacks the window function you used."]),
    ("Implement Dijkstra's algorithm in Python using a heap, then explain why it fails with "
     "negative edge weights and what to use instead.",
     []),
    ("Write a bash script that finds all files over 100MB modified in the last 7 days, prints "
     "them sorted by size with human-readable sizes, and handles filenames with spaces correctly.",
     ["Convert it to PowerShell."]),
    ("Explain the difference between processes and threads, then show a Python example where "
     "multiprocessing beats threading and one where threading is the right choice.",
     ["Where does asyncio fit into this picture?"]),
    ("Review this C function for bugs and undefined behavior:\n\n"
     "```c\nchar *concat(const char *a, const char *b) {\n  char buf[256];\n"
     "  strcpy(buf, a);\n  strcat(buf, b);\n  return buf;\n}\n```\n\n"
     "List every problem and write a corrected version.",
     []),
    ("Write a Python decorator that retries a function up to n times with exponential backoff and "
     "jitter, re-raising the last exception. Type-annotate it properly.",
     ["Extend it to support async functions transparently."]),
    ("Given a CSV with columns date, store, product, units, price, write pandas code to compute "
     "monthly revenue per store, the top product per store, and a pivot table of stores x months.",
     []),
    ("Explain git rebase vs merge with diagrams (ASCII), when to use each, and how to recover "
     "from a rebase gone wrong using reflog.",
     ["My teammate force-pushed over my changes. Walk me through recovery step by step."]),
    ("Implement a simple Markdown-to-HTML converter in Python supporting headers, bold, italics, "
     "links, code spans and unordered lists. No external libraries.",
     ["Add support for fenced code blocks and nested lists."]),

    # --- Science ---
    ("Explain why the sky is blue but sunsets are red, at a level suitable for a curious "
     "12-year-old, then again for a physics undergraduate.",
     ["The undergraduate asks: why isn't the sky violet, since violet light scatters even more?"]),
    ("Walk through what happens inside a lithium-ion battery during charge and discharge, "
     "including the role of the electrolyte and why fast charging degrades capacity.",
     ["Compare with solid-state batteries: what actually changes and what problems remain?"]),
    ("Explain CRISPR-Cas9 gene editing: the natural bacterial mechanism, how it was adapted as a "
     "tool, and what off-target effects are.",
     ["What is base editing and how does it differ from a double-strand break approach?"]),
    ("Describe the life cycle of a massive star from protostar to supernova, and explain why "
     "iron is the end of the line for fusion.",
     ["What determines whether the remnant is a neutron star or a black hole?"]),
    ("Explain plate tectonics evidence: seafloor spreading, magnetic striping, earthquake "
     "distribution, and fossil correlation across continents.",
     []),
    ("Why does ice float, and why is that weird for a solid? Explain hydrogen bonding and the "
     "density anomaly of water, and what a lake would look like each winter if ice sank.",
     []),
    ("Explain how mRNA vaccines work step by step, from injection to immune memory, and why the "
     "lipid nanoparticle matters.",
     ["How does this differ from viral-vector and protein-subunit vaccines?"]),
    ("Explain the greenhouse effect quantitatively: solar constant, albedo, black-body radiation, "
     "and derive the Earth's temperature with and without an atmosphere.",
     ["Where does the logarithmic dependence of forcing on CO2 concentration come from?"]),
    ("What is quantum entanglement, what does Bell's theorem actually rule out, and why can't "
     "entanglement be used to send information faster than light?",
     []),
    ("Explain how a neuron fires: resting potential, ion channels, action potential propagation, "
     "and synaptic transmission. Include the actual ion concentrations.",
     ["How do local anesthetics like lidocaine interfere with this process?"]),

    # --- Medicine / health ---
    ("Explain the difference between type 1 and type 2 diabetes: causes, mechanisms, typical "
     "onset, and treatment approaches.",
     ["What is insulin resistance at the cellular level?"]),
    ("A friend says antibiotics don't work on colds. Explain why they're right, what antibiotics "
     "actually do, and how antibiotic resistance evolves.",
     []),
    ("Describe what happens physiologically during aerobic exercise: heart rate, stroke volume, "
     "oxygen delivery, and the shift between energy systems over a 60-minute run.",
     ["Explain lactate threshold and how athletes train to raise it."]),
    ("Explain how common painkillers differ: paracetamol, ibuprofen, and aspirin — mechanisms, "
     "risks, and when each is preferred.",
     []),

    # --- History / geography ---
    ("Trace the causes of World War I from the 1870s onward: alliance systems, imperial "
     "competition, the arms race, and the July Crisis. Where was the last realistic off-ramp?",
     ["How did the war's ending set up conditions for World War II?"]),
    ("Compare the Roman Republic's fall with the transition from Weimar Germany: institutional "
     "erosion, political violence, and the role of individual actors. Where does the analogy break?",
     []),
    ("Explain the Silk Road: main routes, what actually moved along them (goods, ideas, "
     "diseases), and why it declined.",
     ["How did maritime trade routes change the equation after 1500?"]),
    ("Why is the Nile's annual flood central to ancient Egyptian civilization, and how did the "
     "Aswan High Dam change Egypt's agriculture and ecology?",
     []),
    ("Describe the Meiji Restoration: what triggered it, the key reforms, and how Japan "
     "industrialized in one generation.",
     ["Contrast with China's Self-Strengthening Movement in the same period: why the different outcomes?"]),
    ("Explain how the Himalayas formed, why they're still rising, and their effect on Asian "
     "climate including the monsoon.",
     []),

    # --- Economics / finance / business ---
    ("Summarize the key arguments for and against rent control, citing the standard economic "
     "models involved, and give your overall assessment.",
     ["Steelman the side your assessment went against."]),
    ("Explain inflation: demand-pull vs cost-push, the role of central banks, and why moderate "
     "inflation is targeted instead of zero.",
     ["Walk through the 1970s stagflation and what it taught policymakers."]),
    ("Explain compound interest with a concrete retirement example: saving $500/month at 7% "
     "from age 25 vs starting at 35. Show the math and a year-by-year table for the first 5 years.",
     []),
    ("What is comparative advantage? Work a two-country, two-goods numeric example where one "
     "country is worse at producing both goods yet trade still benefits both.",
     ["What real-world frictions does the model ignore, and when do they dominate?"]),
    ("Write a one-page business plan for a mobile bicycle repair service in a mid-size city: "
     "target market, pricing, unit economics, and the three biggest risks.",
     ["Draft a cold outreach email to apartment building managers proposing a partnership."]),
    ("Explain options basics: calls, puts, covered calls and protective puts, with payoff "
     "diagrams described in text and a concrete example using a $50 stock.",
     []),

    # --- Philosophy / ethics / society ---
    ("Explain the trolley problem and its main variants, what each variant is designed to probe, "
     "and how consequentialists and deontologists respond.",
     ["How do these intuitions map onto real autonomous-vehicle design decisions?"]),
    ("Compare Stoicism and Epicureanism as practical life philosophies: what each actually "
     "recommends day to day, and where they agree more than people assume.",
     []),
    ("What is the Ship of Theseus problem, and how do different theories of identity answer it? "
     "Apply each answer to a band with no original members and a company after full turnover.",
     []),
    ("Present the strongest argument for and against the claim that free will is compatible with "
     "determinism, then explain which premises the disagreement actually hinges on.",
     []),
    ("Explain Rawls' veil of ignorance and the difference principle, then apply the framework to "
     "designing a tax system.",
     ["What is Nozick's strongest objection to this framework?"]),

    # --- Creative writing ---
    ("Write a 400-word short story about a lighthouse keeper who discovers the light attracts "
     "something other than ships, with a twist ending.",
     ["Retell the same story as a series of terse logbook entries."]),
    ("Write a poem about a city at 4 a.m. in strict iambic pentameter, 16 lines, ABAB rhyme "
     "scheme. Then annotate the scansion of the first quatrain.",
     []),
    ("Write the opening scene of a screenplay: two estranged siblings meet at a storage unit "
     "auction where their late mother's unit is up for bid. Proper screenplay format.",
     ["Continue the scene, but a stranger outbids them and offers a deal."]),
    ("Write a product description for a fictional device: a alarm clock that wakes you with the "
     "smell of coffee. Three versions: luxury brand voice, budget brand voice, and absurdist.",
     []),
    ("Write a villanelle about forgetting a language you once spoke fluently.",
     ["Now explain the poem's imagery choices as if annotating it for a literature class."]),
    ("Tell a complete story in exactly 10 sentences, each sentence exactly one word longer than "
     "the previous, starting with a one-word sentence.",
     []),
    ("Write a dialogue between a medieval blacksmith and a time traveler trying to explain what "
     "a website is, without using any technical vocabulary the blacksmith wouldn't know.",
     []),
    ("Write song lyrics for a folk ballad about a container ship stuck sideways in a canal, "
     "verse-chorus-verse-chorus-bridge-chorus structure.",
     []),

    # --- Professional / practical writing ---
    ("Write a resignation letter for a software engineer leaving on good terms after 6 years, "
     "then a version for leaving after a dispute over unpaid overtime — professional but firm.",
     []),
    ("Draft a press release for a small robotics startup announcing a $12M Series A round, "
     "including quotes from the CEO and lead investor.",
     ["Now write the internal all-hands announcement of the same news, different tone."]),
    ("Rewrite this passage to be clear and concise, then explain your edits: 'Due to the fact "
     "that the implementation of the aforementioned system has been delayed as a result of "
     "unforeseen circumstances, it is our recommendation that stakeholders should consider the "
     "utilization of alternative solutions in the interim period until such time as the system "
     "becomes operational.'",
     []),
    ("Write a performance review for a fictional employee who is technically excellent but "
     "dismissive in code reviews: specific, fair, with concrete growth actions.",
     []),
    ("Create a detailed agenda for a 90-minute quarterly planning meeting for an 8-person team, "
     "with timeboxes, pre-reads, and decision points.",
     []),

    # --- Data / structured output ---
    ("Generate a JSON object describing a fictional bookstore inventory: 8 books with nested "
     "author objects, genres arrays, stock info, and prices. Then write a JSON Schema it validates against.",
     []),
    ("Create a YAML CI pipeline config for a Python project: lint, type-check, test on three "
     "Python versions, build a wheel, and publish on tags. Annotate each section with comments.",
     []),
    ("Here is sales data: Q1 north 120, Q1 south 95, Q2 north 140, Q2 south 88, Q3 north 133, "
     "Q3 south 102, Q4 north 155, Q4 south 121. Present it as a markdown table with totals and "
     "growth rates, then describe three insights.",
     []),
    ("Design a REST API for a recipe-sharing app: resources, endpoints, methods, status codes, "
     "and example request/response JSON for the three most important endpoints.",
     ["How would you version this API and handle breaking changes?"]),
    ("Convert this prose into a decision table and a flowchart (described in mermaid syntax): "
     "'Orders over $100 ship free unless international. International orders under $100 pay $25 "
     "shipping, over $100 pay $10. Express doubles any shipping cost but is free for members "
     "on domestic orders over $100.'",
     []),

    # --- Summarization / analysis of provided text ---
    ("Summarize the following in three bullet points, then critique its argument: 'Remote work "
     "has permanently changed the office. Companies clinging to five-day office mandates are "
     "fighting the last war. The data shows productivity held steady or rose during remote "
     "years, and the talent market now prices flexibility as compensation. Offices will survive, "
     "but as collaboration spaces, not attendance theaters. The firms that win the next decade "
     "will be the ones that measure output, not presence.'",
     ["Write the strongest one-paragraph rebuttal from a CEO who disagrees."]),
    ("Explain what this analogy gets right and wrong: 'DNA is like a blueprint for the body.'",
     ["Propose two better analogies and their own failure modes."]),

    # --- Casual chat / advice / roleplay ---
    ("I've been offered a promotion to manager but I love hands-on engineering work. The pay "
     "bump is 15%. How should I think about this decision?",
     ["I took the job and three months in I miss coding badly. Now what?"]),
    ("My sourdough starter smells like acetone and barely rises. Diagnose the likely causes and "
     "give me a recovery plan for the next five days.",
     []),
    ("Roleplay as a grumpy but knowledgeable hardware store employee. I come in saying my "
     "bathroom faucet drips. Stay in character and help me fix it.",
     ["I replaced the washer like you said but now it drips worse."]),
    ("I have a job interview for a data analyst role on Thursday. Run a mock interview: ask me "
     "one question at a time, starting with a behavioral one.",
     ["Give me tough but fair feedback on this answer: 'I once found an error in a report and told my boss about it.'"]),
    ("Plan a 10-day trip through Japan for two people who like hiking and food but hate crowds, "
     "in late November. Include a day-by-day itinerary.",
     ["One traveler sprains an ankle on day 4. Rework the remaining days."]),
    ("My 6-year-old asked me why people die. Give me a few honest, age-appropriate ways to "
     "answer, for a secular household, and what to avoid saying.",
     []),
    ("Settle a debate: my partner says you should rinse pasta after cooking, I say never. Who's "
     "right and why, and are there exceptions?",
     []),
    ("I want to start running but I've been sedentary for years and I'm 40. Give me a realistic "
     "8-week couch-to-5k plan with injury red flags to watch for.",
     []),

    # --- Cooking / food ---
    ("Explain the Maillard reaction and how it differs from caramelization, then give five "
     "practical kitchen techniques that exploit it.",
     []),
    ("Design a week of dinners for a family of four where one member is vegetarian and one is "
     "gluten-free, sharing as many components as possible. Include a consolidated shopping list.",
     []),
    ("Why does my homemade mayonnaise keep breaking? Explain the emulsion science and give a "
     "foolproof rescue method for a broken batch.",
     []),

    # --- Sports / games / music ---
    ("Explain the offside rule in football (soccer) precisely, including the tricky cases: "
     "level positions, deflections, goalkeepers, and passive offside.",
     []),
    ("Analyze the classic Queen's Gambit Declined: main line moves 1-7 with the ideas behind "
     "each move, common traps, and typical middlegame plans for both sides.",
     ["Show a famous game in this opening and annotate the turning point."]),
    ("Explain why a 12-bar blues in E works harmonically: the chords, the role of dominant "
     "sevenths, blue notes, and why the V-IV-I turnaround resolves satisfyingly.",
     []),
    ("Design a balanced D&D encounter for four level-5 players in a flooded temple, with "
     "terrain effects, an interesting enemy composition, and a non-combat resolution path.",
     []),

    # --- DIY / practical ---
    ("My interior door won't latch — the bolt misses the strike plate by about 3mm vertically. "
     "Walk me through diagnosis and the three standard fixes from least to most invasive.",
     []),
    ("Explain how to safely jump-start a car with cables, the correct connection order and why, "
     "and the cases where you should not attempt it.",
     []),
    ("I want to build a raised garden bed, 3m x 1.2m, on a slight slope. Materials list, cost "
     "estimate, step-by-step build, and what soil mix to fill it with.",
     []),

    # --- Technology explainers ---
    ("Explain how HTTPS actually works when I visit a website: DNS, TCP, TLS handshake, "
     "certificates, and what the padlock does and does not guarantee.",
     ["What changes with TLS 1.3 and why were older versions deprecated?"]),
    ("Explain how GPS determines position: satellites, time signals, trilateration, why "
     "relativity corrections are needed, and typical error sources.",
     []),
    ("What actually happens from pressing the power button to seeing a login screen? BIOS/UEFI, "
     "bootloader, kernel init, services — explain the chain for a modern Linux system.",
     []),
    ("Explain how noise-cancelling headphones work, the difference between passive and active "
     "cancellation, and why they struggle with speech but excel with engine drone.",
     []),
    ("Explain large language model inference to a software engineer who knows nothing about ML: "
     "tokens, the transformer loop, KV cache, and why long contexts get slow.",
     ["Now explain quantization of such models and its quality/speed tradeoffs."]),

    # --- Translation / linguistics ---
    ("Translate this into French, German and Japanese, preserving the informal tone, and note "
     "one interesting translation choice for each: 'Honestly, the meeting could have been an "
     "email, but at least the snacks were good.'",
     []),
    ("Explain grammatical gender to a native English speaker: what it is, what it isn't, how "
     "it works differently in Spanish, German and Swahili, and common misconceptions.",
     []),
    ("What is the difference between a dialect and a language? Discuss with examples from "
     "Chinese, Arabic, and Scandinavian languages, and explain the sociolinguistic factors.",
     []),
    ("Break down the etymology of ten common English words with surprising origins, presented "
     "as a table: word, origin language, original meaning, path into English.",
     []),

    # --- Chinese ---
    ("请解释一下什么是区块链技术,它的核心原理是什么,以及除了加密货币之外还有哪些实际应用场景?",
     ["请针对供应链溯源这个应用,详细说明区块链具体解决了什么传统方案解决不了的问题。"]),
    ("我想学做正宗的麻婆豆腐,请给出详细的食材清单和步骤,并解释'麻'和'辣'分别来自哪里,如何平衡。",
     []),
    ("请以'秋雨'为题写一首七言绝句,并解释其中的意象和平仄安排。",
     ["再用同样的主题写一首现代自由诗,比较两种形式的表达差异。"]),
    ("小明有一些苹果,如果每天吃3个,可以多吃4天;如果每天吃5个,就少吃4天。问小明有多少个苹果?请列方程详细求解。",
     []),
    ("请帮我写一封正式的商务邮件:我们公司需要向长期合作的供应商提出降价10%的请求,理由是采购量将翻倍。语气要礼貌但坚定。",
     []),

    # --- Japanese ---
    ("初めて京都を訪れる外国人の友人に、3日間のおすすめプランを教えてください。有名な観光地だけでなく、地元の人が好む場所も含めてください。",
     ["雨の日の代替プランも提案してください。"]),
    ("取引先に納期延期をお願いするビジネスメールを書いてください。理由は部品調達の遅れで、2週間の延期が必要です。適切な敬語を使ってください。",
     []),
    ("再帰関数とは何ですか?プログラミング初心者にもわかるように、日常の例えを使って説明してから、Pythonで階乗を計算する例を見せてください。",
     []),
    ("「もったいない」という言葉の意味と文化的背景を説明し、他の言語に翻訳しにくい日本語の言葉を5つ挙げて、それぞれ解説してください。",
     []),

    # --- Korean ---
    ("한국의 전통 발효 음식인 김치의 발효 과정을 과학적으로 설명해 주세요. 유산균의 역할과 온도가 맛에 미치는 영향도 포함해 주세요.",
     []),
    ("처음 서울에 오는 친구에게 대중교통 이용법을 알려주세요. 티머니 카드, 지하철, 버스 환승 시스템을 포함해서 설명해 주세요.",
     []),
    ("이력서에 쓸 자기소개서를 도와주세요. 저는 5년 경력의 마케팅 전문가이고 데이터 분석 역량이 강점입니다. 300자 정도로 작성해 주세요.",
     ["좀 더 구체적인 성과 중심으로 다시 써 주세요. 예를 들어 수치를 포함해서요."]),

    # --- Spanish ---
    ("Explica la diferencia entre 'ser' y 'estar' para un estudiante de español, con reglas "
     "claras, ejemplos y los casos donde ambos son posibles pero cambian el significado.",
     []),
    ("Escribe una receta detallada de paella valenciana auténtica, con los ingredientes que "
     "nunca deben faltar y los errores más comunes que cometen los principiantes.",
     ["¿Qué diferencias hay entre la paella valenciana y la paella de mariscos?"]),
    ("Resume la historia de la independencia de los países latinoamericanos en el siglo XIX: "
     "causas comunes, figuras principales y por qué se fragmentó en tantas naciones.",
     []),
    ("Mi hijo de 8 años le tiene miedo a la oscuridad y no quiere dormir solo. ¿Qué estrategias "
     "recomiendan los psicólogos infantiles? Dame un plan práctico para dos semanas.",
     []),

    # --- French ---
    ("Expliquez la différence entre le passé composé et l'imparfait, avec des exemples où le "
     "choix du temps change complètement le sens de la phrase.",
     []),
    ("Rédigez une lettre de motivation pour un poste d'ingénieur logiciel dans une startup "
     "parisienne, pour un candidat avec 3 ans d'expérience qui veut quitter un grand groupe.",
     []),
    ("Quelle est la recette traditionnelle du bœuf bourguignon ? Détaillez les étapes, le choix "
     "du vin, et expliquez pourquoi la cuisson lente transforme la viande.",
     []),
    ("Expliquez le fonctionnement de l'Union européenne à un lycéen : les institutions "
     "principales, comment une loi est adoptée, et le débat sur la souveraineté.",
     ["Quels sont les arguments économiques pour et contre l'euro comme monnaie unique ?"]),

    # --- German ---
    ("Erkläre den Unterschied zwischen 'kennen' und 'wissen' für Deutschlernende, mit Beispielen "
     "und den häufigsten Fehlern von Englischsprachigen.",
     []),
    ("Schreibe eine formelle E-Mail an die Hausverwaltung: Die Heizung in der Wohnung funktioniert "
     "seit einer Woche nicht, mehrere Anrufe blieben erfolglos, und du kündigst eine Mietminderung an.",
     []),
    ("Erkläre das duale Ausbildungssystem in Deutschland: wie es funktioniert, warum es im "
     "Ausland als Vorbild gilt, und welche Schwächen es hat.",
     []),
    ("Was ist der Unterschied zwischen Bundesrat und Bundestag? Erkläre die Gesetzgebung in "
     "Deutschland Schritt für Schritt an einem konkreten Beispiel.",
     []),

    # --- Russian ---
    ("Объясните разницу между глаголами совершенного и несовершенного вида в русском языке, с "
     "примерами, где выбор вида полностью меняет смысл предложения.",
     []),
    ("Напишите подробный рецепт настоящего борща с пошаговыми инструкциями. Объясните, зачем "
     "свёклу готовят отдельно и как добиться насыщенного цвета.",
     []),
    ("Расскажите о творчестве Достоевского для человека, который никогда его не читал: с какого "
     "романа начать, основные темы и почему он до сих пор актуален.",
     []),

    # --- Portuguese ---
    ("Explique a diferença entre o português do Brasil e o de Portugal: pronúncia, vocabulário, "
     "gramática e alguns mal-entendidos engraçados entre falantes.",
     []),
    ("Escreva um roteiro de 5 dias no Rio de Janeiro para quem gosta de natureza e cultura, "
     "evitando as armadilhas turísticas mais caras. Inclua dicas de segurança realistas.",
     []),
    ("Explique como funciona a declaração de imposto de renda para uma pessoa que acabou de "
     "começar seu primeiro emprego formal. O que é retido na fonte e o que é restituição?",
     []),

    # --- Italian ---
    ("Spiega la differenza tra i vari tipi di pasta e i sughi che tradizionalmente li "
     "accompagnano: perché le forme contano, con esempi concreti e gli abbinamenti da evitare.",
     []),
    ("Scrivi una guida di due giorni a Firenze per amanti dell'arte: itinerario, come evitare "
     "le code agli Uffizi, e tre luoghi meno conosciuti che meritano una visita.",
     []),

    # --- Arabic ---
    ("اشرح الفرق بين اللغة العربية الفصحى واللهجات العامية، مع أمثلة من اللهجة المصرية والشامية "
     "والخليجية، ولماذا يفهم العرب الفصحى جميعاً بينما قد لا يفهمون لهجات بعضهم البعض.",
     []),
    ("اكتب وصفة تفصيلية لتحضير الكبسة السعودية التقليدية، مع شرح التوابل الأساسية وسر الأرز "
     "المفلفل الذي لا يتكتل.",
     []),

    # --- Hindi ---
    ("भारत में मानसून कैसे बनता है? इसकी प्रक्रिया को विस्तार से समझाइए और यह भी बताइए कि यह "
     "भारतीय कृषि और अर्थव्यवस्था के लिए इतना महत्वपूर्ण क्यों है।",
     []),
    ("दाल मखनी बनाने की पूरी विधि बताइए। कौन सी दाल इस्तेमाल होती है, कितनी देर पकानी चाहिए, "
     "और रेस्टोरेंट जैसा स्वाद घर पर कैसे लाया जाए?",
     []),

    # --- Turkish ---
    ("Türk kahvesinin geleneksel hazırlanışını adım adım anlatır mısın? Köpüğün sırrı nedir ve "
     "fal bakma geleneği nereden geliyor?",
     []),
    ("İstanbul'u ilk kez ziyaret edecek birine 4 günlük bir gezi planı hazırla. Hem Avrupa hem "
     "Anadolu yakasını kapsasın, ulaşım önerileriyle birlikte.",
     []),

    # --- Dutch ---
    ("Leg uit waarom Nederland zo goed is in watermanagement: de Deltawerken, polders en wat "
     "andere landen kunnen leren van de Nederlandse aanpak van zeespiegelstijging.",
     []),

    # --- Polish ---
    ("Wyjaśnij odmianę przez przypadki w języku polskim osobie anglojęzycznej: po co jest "
     "siedem przypadków, z przykładami pokazującymi jak zmienia się znaczenie.",
     []),

    # --- Swedish ---
    ("Förklara begreppet 'lagom' och andra svenska ord som är svåra att översätta. Hur speglar "
     "de svensk kultur och samhälle?",
     []),

    # --- Vietnamese ---
    ("Hãy giải thích cách nấu phở bò truyền thống: nước dùng cần những gia vị gì, ninh xương "
     "bao lâu, và bí quyết để nước dùng trong mà vẫn đậm đà.",
     []),

    # --- Thai ---
    ("ช่วยอธิบายวิธีทำต้มยำกุ้งแบบดั้งเดิม ต้องใช้สมุนไพรอะไรบ้าง และความแตกต่างระหว่างต้มยำน้ำใสกับน้ำข้นคืออะไร",
     []),

    # --- Indonesian ---
    ("Jelaskan mengapa Indonesia memiliki begitu banyak gunung berapi, bagaimana Cincin Api "
     "Pasifik bekerja, dan bagaimana masyarakat hidup berdampingan dengan risiko letusan.",
     []),

    # --- Finnish ---
    ("Selitä suomalainen saunakulttuuri ulkomaalaiselle: saunan merkitys, käyttäytymissäännöt ja "
     "miksi sauna on niin tärkeä osa suomalaista identiteettiä.",
     []),

    # --- Mixed / code-switching / edge formats ---
    ("Explain the rules of cricket to an American baseball fan, mapping each concept to its "
     "closest baseball equivalent where one exists, in a comparison table.",
     []),
    ("Write the same error message in five different software voices: a 1980s mainframe, "
     "a modern minimalist app, an over-friendly startup, a legal-compliance system, and a pirate.",
     []),
    ("Compose a formal proof, a limerick, and a tweet all explaining why 0.999... equals 1.",
     []),
    ("Interview yourself: write five hard questions a skeptical journalist would ask an AI "
     "assistant about its limitations, and answer them honestly.",
     []),
    ("Create a glossary of 15 terms a new employee at a semiconductor fab would need to know, "
     "ordered by how soon they'll encounter each, with two-sentence definitions.",
     []),
]


col_default = "[0m"
col_yellow = "[33;1m"
col_blue = "[34;1m"
col_green = "[32;1m"


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
        max_chunk_size = 2048,
    )

    TEMPLATE_VARS = DEFAULT_TEMPLATE_VARS.copy()
    TEMPLATE_VARS.update(args.template_vars)

    # Some templates reject specific variables (e.g. qwen3.8 only accepts reasoning_effort
    # xhigh/medium/low, and errors on the qbench default "high"). Drop the offending key(s) and
    # fall back to the template's own defaults rather than crashing a long run
    probe = [{"role": "user", "content": "hi"}]
    def template_works(vars_):
        try:
            tokenizer.hf_chat_template(probe, add_generation_prompt = True, **vars_)
            return True
        except Exception:
            return False
    while TEMPLATE_VARS and not template_works(TEMPLATE_VARS):
        for k in list(TEMPLATE_VARS):
            if template_works({kk: v for kk, v in TEMPLATE_VARS.items() if kk != k}):
                print(f" !! Chat template rejects {k} = {TEMPLATE_VARS[k]!r}; using template default")
                del TEMPLATE_VARS[k]
                break
        else:
            print(" !! Chat template rejects all variable combinations; using template defaults")
            TEMPLATE_VARS = {}

    target_tokens = args.target_tokens or args.cal_rows * args.cal_cols
    rows = []
    total_in = total_out = 0

    def total_packed():
        return total_in + total_out

    with ProgressBar("Sampling", target_tokens) as pb:
        for epoch in range(args.max_epochs):
            if total_packed() >= target_tokens:
                break
            convs = [{"messages": [], "turns": [seed] + fu, "alive": True}
                     for seed, fu in CONVERSATIONS]
            max_turns = max(len(c["turns"]) for c in convs)

            # Turn-major waves: every conversation's turn t runs as one concurrent batch, so a
            # small token target still spreads across all domains, and follow-up turns only need
            # the previous wave's responses
            for t_idx in range(max_turns):
                if total_packed() >= target_tokens:
                    break
                pending = {}    # c_idx -> {"input_ids", "chunks", "eos_reason"}
                for c_idx, conv in enumerate(convs):
                    if not conv["alive"] or t_idx >= len(conv["turns"]):
                        continue
                    conv["messages"].append({"role": "user", "content": conv["turns"][t_idx]})
                    input_ids = tokenizer.hf_chat_template(
                        conv["messages"], add_generation_prompt = True, **TEMPLATE_VARS)
                    job = Job(
                        input_ids = input_ids,
                        max_new_tokens = args.max_new_tokens,
                        stop_conditions = config.eos_token_id_list,
                        decode_special_tokens = True,
                        identifier = c_idx,
                        seed = zlib.crc32(f"{args.seed}|{epoch}|{c_idx}|{t_idx}".encode()),
                    )
                    generator.enqueue(job)
                    pending[c_idx] = {"input_ids": input_ids, "chunks": [], "eos_reason": None}

                print(f" -- Epoch {epoch}, turn {t_idx}: {col_blue}{len(pending)}{col_default} jobs")
                wave_tokens = 0
                while generator.num_remaining_jobs():
                    for result in generator.iterate():
                        if result["stage"] != "streaming":
                            continue
                        p = pending[result["identifier"]]
                        if "token_ids" in result:
                            p["chunks"].append(result["token_ids"])
                        if result["eos"]:
                            p["eos_reason"] = result.get("eos_reason")
                            n_new = sum(c.shape[-1] for c in p["chunks"])
                            wave_tokens += n_new + p["input_ids"].shape[-1]
                            print(f"    conv {result['identifier']:3} turn {t_idx}: "
                                  f"{col_yellow}{n_new:5}{col_default} tokens "
                                  f"({p['eos_reason']})   total "
                                  f"{col_green}{total_packed() + wave_tokens:8,}{col_default}")
                    pb.update(min(total_packed() + wave_tokens, target_tokens))

                for c_idx, p in pending.items():
                    conv = convs[c_idx]
                    chunks = p["chunks"]
                    response_ids = torch.cat(chunks, dim = -1)[0] if chunks \
                        else torch.empty(0, dtype = torch.long)
                    if response_ids.numel() == 0:
                        conv["messages"].pop()
                        conv["alive"] = False
                        continue
                    rows.append({
                        "conversation": c_idx,
                        "turn": t_idx,
                        "epoch": epoch,
                        "eos_reason": p["eos_reason"],
                        "input_ids": p["input_ids"][0].tolist(),
                        "response_ids": response_ids.tolist(),
                    })
                    total_in += p["input_ids"].shape[-1]
                    total_out += response_ids.numel()
                    conv["messages"].append({
                        "role": "assistant",
                        "content": clean_response_for_context(tokenizer, response_ids),
                    })
                pb.update(min(total_packed(), target_tokens))

    if total_packed() < target_tokens:
        print(f" !! Collected only {total_packed():,} of {target_tokens:,} target tokens after "
              f"{args.max_epochs} epoch(s); consider more epochs or a higher --max_new_tokens")

    # qbench-compatible trace
    out = {
        "model": args.model_dir,
        "vocab_size": tokenizer.actual_vocab_size,
        "template_vars": TEMPLATE_VARS,
        "meta": {"rows": len(rows), "input_tokens": total_in, "output_tokens": total_out},
        "rows": rows,
    }
    with open(args.output, "w") as f:
        json.dump(out, f)
    print(f" -- {len(rows)} trace rows, {total_in:,} context + {total_out:,} response tokens "
          f"-> {args.output}")

    # Packed calibration rows: each trace row's exact model-visible stream (templated context +
    # sampled response), row order shuffled for mixing, concatenated and sliced to fixed width
    cal_out = args.cal_out or os.path.splitext(args.output)[0] + ".safetensors"
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    stream = []
    for i in order:
        stream += rows[i]["input_ids"] + rows[i]["response_ids"]
    n_rows = len(stream) // args.cal_cols
    if args.cal_rows and n_rows > args.cal_rows:
        n_rows = args.cal_rows
    packed = torch.tensor(stream[:n_rows * args.cal_cols], dtype = torch.long)
    packed = packed.view(n_rows, args.cal_cols)
    save_file({"input_ids": packed}, cal_out)
    print(f" -- {n_rows} calibration rows x {args.cal_cols} tokens -> {cal_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev = False)
    model_init.add_args(parser, default_cache_size = 131072, add_draft_model_args = True)
    parser.add_argument("-o", "--output", type = str, required = True, help = "Output trace (JSON, qbench-compatible)")
    parser.add_argument("-co", "--cal_out", type = str, default = None, help = "Output packed calibration rows (safetensors), default: derived from --output")
    parser.add_argument("-cr", "--cal_rows", type = int, default = 250, help = "Calibration rows to pack, default: 250")
    parser.add_argument("-cc", "--cal_cols", type = int, default = 2048, help = "Tokens per calibration row, default: 2048")
    parser.add_argument("--target_tokens", type = int, default = None, help = "Token collection target, default: cal_rows * cal_cols")
    parser.add_argument("--max_new_tokens", type = int, default = 3072, help = "Per-turn generation cap, default: 3072")
    parser.add_argument("--max_epochs", type = int, default = 4, help = "Max passes over the seed set (later epochs resample with different seeds), default: 4")
    parser.add_argument("--seed", type = int, default = 0, help = "Base RNG seed (per-job seeds derive from it)")
    parser.add_argument("-tv", "--template_vars", type = json.loads, default = {}, help = 'JSON dict of chat template variables, merged over the defaults')
    main(parser.parse_args())
