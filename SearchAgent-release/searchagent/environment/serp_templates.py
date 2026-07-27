"""Rich SERP templates for all search scenarios.

Each scenario has 10+ organic results and 3-4 ads, covering:
- Product/commercial queries (headphones, shoes, laptops, flights, credit cards)
- Informational queries (tutorials, health, education)
- Local queries (restaurants)
- Financial/insurance queries

Templates also cover reformulated variants so the LLM-generated
queries still produce realistic SERPs.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# Type alias: (title, snippet, url)
ResultTemplate = Tuple[str, str, str]


ORGANIC_TEMPLATES: Dict[str, List[ResultTemplate]] = {

    # ── Scenario 1: Noise Cancelling Headphones ──────────────────────────
    "best noise cancelling headphones": [
        ("Sony WH-1000XM5 Review - Best Overall ANC Headphones",
         "Detailed review of the Sony WH-1000XM5. Industry-leading noise cancellation, "
         "30-hour battery life, multipoint Bluetooth. Rated 9.2/10 by our audio engineers.",
         "www.techradar.com/reviews/sony-wh1000xm5"),
        ("Bose QuietComfort Ultra Headphones - Full Review",
         "Premium adaptive noise cancelling with spatial audio. CustomTune technology "
         "personalizes sound to your ears. Comfortable for all-day wear. $429.",
         "www.bose.com/headphones/quietcomfort-ultra"),
        ("The 10 Best Noise-Cancelling Headphones of 2024 | Wirecutter",
         "After 200+ hours of testing 53 headphones, we pick the best for every budget. "
         "Top choice: Sony XM5. Budget pick: Sony WH-CH720N ($98). Runner-up: Bose QC Ultra.",
         "www.wirecutter.com/reviews/best-noise-cancelling-headphones"),
        ("Apple AirPods Max vs Sony WH-1000XM5: Which Should You Buy?",
         "Head-to-head comparison on noise cancellation, sound quality, comfort, battery, "
         "and price. Sony wins on value; AirPods Max wins on build quality and Apple ecosystem.",
         "www.cnet.com/tech/mobile/airpods-max-vs-sony-xm5-compared"),
        ("Budget Noise-Cancelling Headphones Under $100 That Actually Work",
         "You don't need to spend $350+ for effective ANC. The Anker Soundcore Space Q45 "
         "($79) and JBL Tune 770NC ($69) offer surprisingly good noise cancellation.",
         "www.soundguys.com/best-budget-noise-cancelling-headphones"),
        ("Sennheiser Momentum 4 Wireless Review: Audiophile ANC",
         "German audio engineering with 60-hour battery. Adaptive ANC with transparency mode. "
         "Superior sound quality compared to Sony/Bose but slightly bulkier design.",
         "www.whathifi.com/reviews/sennheiser-momentum-4-wireless"),
        ("How Active Noise Cancellation Technology Works - Explained",
         "ANC uses external microphones to sample ambient sound and generates anti-phase "
         "signals to cancel it. Feed-forward vs feedback vs hybrid designs compared.",
         "www.howstuffworks.com/gadgets/audio-music/noise-cancelling-headphone.htm"),
        ("r/headphones - Best ANC headphones for daily commuting in 2024?",
         "Community thread with 847 comments. Top voted: Sony XM5 for metro/bus, Bose QC45 "
         "for airplane travel. Audiophile pick: Sennheiser Momentum 4.",
         "www.reddit.com/r/headphones/comments/abc123/best_anc_commuting"),
        ("JBL Tour One M2 Review: Best Mid-Range ANC",
         "Adaptive noise cancelling with 50-hour battery life. Spatial sound and head tracking. "
         "Excellent microphone quality for calls. $249 — great value.",
         "www.tomsguide.com/reviews/jbl-tour-one-m2"),
        ("Noise-Cancelling Headphones Buying Guide 2024 | Rtings",
         "Measured data on 400+ headphones. Compare ANC performance, frequency response, "
         "comfort, and portability. Interactive comparison tool with side-by-side charts.",
         "www.rtings.com/headphones/reviews/best/noise-cancelling"),
        ("Jabra Elite 85h Long-Term Review After 2 Years",
         "SmartActive ANC adapts to environment. Excellent call quality with 6 microphones. "
         "Battery still holds 36 hours after 2 years. IP52 rain resistant.",
         "www.pcmag.com/reviews/jabra-elite-85h"),
        ("Marshall Monitor II ANC Review: Style Meets Function",
         "Iconic retro design with competent noise cancellation. Warm, bass-forward sound "
         "signature. 30-hour battery with ANC on. USB-C and 3.5mm jack.",
         "www.headphonecheck.com/marshall-monitor-ii-anc-review"),
    ],

    # ── Scenario 2: Cheap Flights to Tokyo ───────────────────────────────
    "cheap flights to tokyo": [
        ("Cheap Flights to Tokyo (NRT) from $487 | Google Flights",
         "Compare prices across 300+ airlines. Best deals: ANA direct $612, United via SFO "
         "$487, Japan Airlines $658. Prices for round-trip. Flexible date search available.",
         "www.google.com/travel/flights/tokyo"),
        ("Tokyo Flights: 14 Tips to Find the Cheapest Airfare",
         "Book 8-12 weeks ahead for 23% savings. Tuesday departures average 17% cheaper. "
         "Consider flying into Haneda (HND) instead of Narita — closer and sometimes cheaper.",
         "www.skyscanner.com/tips/cheap-flights-to-tokyo"),
        ("Best Time to Fly to Tokyo: Month-by-Month Price Guide",
         "January and November are cheapest ($480-550 avg). Cherry blossom season (Mar-Apr) "
         "peaks at $850+. Use fare alerts for flash sales. Shoulder seasons save 30-40%.",
         "www.kayak.com/flight-prices/to-tokyo"),
        ("Tokyo Flight Deals & Error Fares | Secret Flying",
         "Current deals: New York to Tokyo $423 RT (ANA), Los Angeles to Tokyo $398 RT "
         "(Zipair). Error fares updated hourly. Sign up for instant alerts.",
         "www.secretflying.com/deals/flights-to-tokyo"),
        ("Zipair: Japan's Budget Airline to Tokyo from $299 One-Way",
         "Low-cost carrier by JAL Group. Routes from LAX, SFO, Honolulu, and Singapore. "
         "Basic fare from $299. Standard fare $449 includes meals and 2 checked bags.",
         "www.zipair.net/en"),
        ("r/travel - How I Flew to Tokyo for $340 Round Trip",
         "Trip report: Used Google Flights flexible dates + mistake fare from Midwest hub. "
         "Comments discuss Scott's Cheap Flights alerts, credit card travel portals, and "
         "positioning flights to catch better deals.",
         "www.reddit.com/r/travel/comments/xyz789/tokyo_340_roundtrip"),
        ("ANA vs JAL: Which Japanese Airline Is Better for Economy?",
         "Detailed comparison of All Nippon Airways and Japan Airlines economy class. Seat "
         "pitch, meal quality, entertainment, and service compared on Tokyo routes.",
         "www.thepointsguy.com/reviews/ana-vs-jal-economy"),
        ("When to Book Flights to Japan: Optimal Booking Window Analysis",
         "Analysis of 2.1M fares shows booking 56 days ahead is optimal for US-Tokyo routes. "
         "Price tracking shows Tuesday 3pm ET has the most fare drops.",
         "www.faredetective.com/optimal-booking-window-tokyo"),
        ("Layover vs Direct Flights to Tokyo: Is the Savings Worth It?",
         "Direct flights average $150-200 more but save 5-8 hours. Seoul (ICN) and Taipei "
         "(TPE) are the best layover airports with comfortable lounges.",
         "www.travelandleisure.com/layover-vs-direct-tokyo"),
        ("Tokyo Travel Guide: What You Need Before Booking Flights",
         "Visa requirements, Japan Rail Pass, Suica card, and pocket WiFi. Book flights "
         "first, then accommodation — hotel prices don't fluctuate as much.",
         "www.lonelyplanet.com/japan/tokyo/planning"),
    ],

    # ── Scenario 3: Python ML Tutorial ───────────────────────────────────
    "python machine learning tutorial": [
        ("Machine Learning with Python Tutorial - Scikit-Learn Guide",
         "Complete beginner tutorial covering supervised/unsupervised learning with Python. "
         "Hands-on projects: spam classifier, house price prediction, image recognition.",
         "www.datacamp.com/tutorial/machine-learning-python"),
        ("Google's Machine Learning Crash Course (Free)",
         "40-hour free course with TensorFlow. Covers linear regression, classification, "
         "neural networks, embeddings, and ML engineering practices. Interactive Colab notebooks.",
         "developers.google.com/machine-learning/crash-course"),
        ("Hands-On Machine Learning with Scikit-Learn and TensorFlow",
         "O'Reilly book by Aurélien Géron. 3rd edition (2023). Covers end-to-end ML projects, "
         "neural networks, deep learning. Code examples on GitHub (47k stars).",
         "www.oreilly.com/library/view/hands-on-machine-learning"),
        ("Python Machine Learning: From Zero to Hero (YouTube Playlist)",
         "24-video series covering NumPy, Pandas, Matplotlib, Scikit-Learn, and PyTorch. "
         "Project-based learning. Each video has downloadable Jupyter notebooks.",
         "www.youtube.com/playlist?list=PLpython-ml-zero-to-hero"),
        ("Fast.ai Practical Deep Learning for Coders",
         "Free course by Jeremy Howard. Top-down approach — build models first, understand "
         "theory later. Covers CNNs, NLP, tabular data, collaborative filtering.",
         "www.fast.ai/courses"),
        ("Kaggle Learn: Intro to Machine Learning (Free Interactive Course)",
         "Learn by doing on Kaggle. 7 lessons with hands-on exercises. Topics: decision "
         "trees, random forests, model validation, underfitting/overfitting.",
         "www.kaggle.com/learn/intro-to-machine-learning"),
        ("The Best Python ML Libraries in 2024: Complete Comparison",
         "Scikit-Learn vs TensorFlow vs PyTorch vs XGBoost vs LightGBM. When to use each, "
         "performance benchmarks, and learning curve comparison.",
         "www.towardsdatascience.com/best-python-ml-libraries-2024"),
        ("Building Your First ML Model in 30 Minutes with Scikit-Learn",
         "Step-by-step tutorial: load Iris dataset, split train/test, train RandomForest, "
         "evaluate with confusion matrix and ROC. Complete code included.",
         "www.realpython.com/first-ml-model-scikit-learn"),
        ("r/learnmachinelearning - Roadmap for Self-Taught ML Engineer",
         "Curated 6-month roadmap: Math foundations (weeks 1-4), Python basics (weeks 5-6), "
         "ML algorithms (weeks 7-12), Deep Learning (weeks 13-18), Projects (weeks 19-24).",
         "www.reddit.com/r/learnmachinelearning/roadmap_self_taught"),
        ("Stanford CS229: Machine Learning - Full Course Notes",
         "Andrew Ng's Stanford course materials. Comprehensive notes on supervised learning, "
         "deep learning, reinforcement learning, and unsupervised learning.",
         "cs229.stanford.edu/main_notes.pdf"),
    ],

    # ── Scenario 4: Running Shoes ────────────────────────────────────────
    "best running shoes 2024": [
        ("The Best Running Shoes of 2024 | Runner's World",
         "Expert-tested guide: Nike Vaporfly 3 (racing), ASICS Gel-Nimbus 26 (cushioned), "
         "Brooks Ghost 16 (daily trainer), Hoka Clifton 9 (max cushion). 42 shoes tested.",
         "www.runnersworld.com/gear/best-running-shoes"),
        ("Nike Pegasus 41 Review: The Perfect Daily Trainer?",
         "Updated React X foam delivers 13% more energy return. Engineered mesh upper for "
         "breathability. 286g men's / 230g women's. Best for 20-50 miles per week. $140.",
         "www.believeintherun.com/nike-pegasus-41-review"),
        ("ASICS Gel-Nimbus 26 vs Brooks Glycerin 21: Cushion King Showdown",
         "Head-to-head comparison of the two most popular max-cushion trainers. Stack heights, "
         "drop, weight, durability, and ride feel compared over 200 miles each.",
         "www.solereview.com/asics-nimbus-26-vs-brooks-glycerin-21"),
        ("Best Marathon Training Shoes: Tested Over 500+ Miles",
         "Long-run specialists: Saucony Endorphin Speed 4, Nike Vomero 17, New Balance "
         "FuelCell Rebel v4. Rotation recommendations for different training phases.",
         "www.irunfar.com/best-marathon-training-shoes"),
        ("How to Choose Running Shoes: Pronation, Cushion & Fit Guide",
         "Understanding gait analysis, pronation types (neutral, over, under), shoe categories "
         "(stability vs neutral vs motion control). When to visit a specialty running store.",
         "www.rei.com/learn/expert-advice/running-shoes.html"),
        ("Hoka Clifton 9 Long-Term Review: 400 Miles Later",
         "Lightweight max cushion shoe (248g). Extended EVA foam with Meta-Rocker geometry. "
         "Excellent for easy runs and recovery days. Outsole durability could be better.",
         "www.roadtrailrun.com/hoka-clifton-9-review"),
        ("Brooks Ghost 16 Review: Still the Best Beginner Running Shoe",
         "DNA Loft v2 foam provides smooth transition. Neutral shoe with broad appeal. "
         "Works for road runs 3-10 miles. Durable to 400+ miles. $140 retail.",
         "www.tomsguide.com/reviews/brooks-ghost-16"),
        ("r/running - What shoes for first marathon? (30+ MPW currently)",
         "Top recommendations: Nike Vomero 17 for daily training, Saucony Endorphin Speed "
         "for tempo/long runs, Nike Vaporfly 3 for race day. Discussion on shoe rotation.",
         "www.reddit.com/r/running/first_marathon_shoes"),
        ("New Balance FuelCell Rebel v4 Review: Speed Without the Price",
         "Lightweight (196g) with FuelCell foam and carbon fiber plate. Snappy for tempo "
         "runs and race day. $139 undercuts Nike Vaporfly significantly.",
         "www.runrepeat.com/new-balance-fuelcell-rebel-v4"),
        ("On Cloudmonster 2 Review: Swiss-Engineered Cloud Comfort",
         "Helion HF foam with CloudTec outsole. Unique rocker design for smooth transitions. "
         "28mm stack height. Best for neutral runners seeking max cushion. $170.",
         "www.outsideonline.com/gear/running/on-cloudmonster-2-review"),
    ],

    # ── Scenario 5: Home Insurance ───────────────────────────────────────
    "home insurance quotes comparison": [
        ("Compare Home Insurance Quotes: Save Up to 35% | NerdWallet",
         "Get personalized quotes from 40+ insurers in 5 minutes. Average savings: $482/year. "
         "Compare coverage, deductibles, and customer ratings side by side.",
         "www.nerdwallet.com/insurance/homeowners-insurance-comparison"),
        ("Best Homeowners Insurance Companies of 2024 | US News",
         "Rankings based on financial strength, customer satisfaction, and claims handling. "
         "#1 Amica Mutual, #2 USAA (military), #3 State Farm. Methodology explained.",
         "www.usnews.com/insurance/homeowners-insurance/best-companies"),
        ("How Much Does Home Insurance Cost? 2024 Average Rates",
         "National average: $2,230/year for $300K dwelling coverage. Varies by state: "
         "Florida ($4,218), Ohio ($1,234), California ($1,890). Factors affecting premiums.",
         "www.bankrate.com/insurance/homeowners-insurance/average-cost"),
        ("6 Ways to Lower Your Home Insurance Premium Without Losing Coverage",
         "Bundle with auto (save 15-25%), raise deductible to $2,500 (save 12%), install "
         "security system (save 5-15%), improve credit score, shop annually.",
         "www.consumerreports.org/homeowners-insurance/lower-premium-tips"),
        ("State Farm vs Allstate vs Progressive: Home Insurance Compared",
         "Side-by-side comparison of coverage options, pricing, claims process, digital "
         "experience, and discounts. State Farm best overall, Progressive best for bundling.",
         "www.policygenius.com/homeowners-insurance/state-farm-vs-allstate-vs-progressive"),
        ("What Does Homeowners Insurance Cover? (And What It Doesn't)",
         "Covers: dwelling, personal property, liability, additional living expenses. "
         "Doesn't cover: floods, earthquakes, maintenance issues, sewer backup (without rider).",
         "www.iii.org/article/what-covered-standard-homeowners-policy"),
        ("Home Insurance Deductible Guide: How to Choose the Right Amount",
         "$500 vs $1,000 vs $2,500 vs percentage deductibles. Higher deductible = lower "
         "premium. Rule: choose the highest you can comfortably pay out-of-pocket.",
         "www.thebalancemoney.com/home-insurance-deductible-guide"),
        ("r/personalfinance - Switching home insurance saved me $700/year",
         "User shares experience switching from Allstate to Erie Insurance. Discussion "
         "covers independent agents, quote comparison tools, and policy review tips.",
         "www.reddit.com/r/personalfinance/home_insurance_saved_700"),
        ("Lemonade Home Insurance Review: AI-Powered Claims in 3 Minutes",
         "Digital-first insurer with instant quotes. Base coverage from $25/month. Pros: fast "
         "claims, modern app. Cons: limited availability, lower coverage limits.",
         "www.investopedia.com/reviews/lemonade-homeowners-insurance"),
        ("Independent Insurance Agent vs Direct: Which Saves More?",
         "Independent agents compare 10-20 carriers vs single-company agents. Average savings "
         "through independent agents: 12-18%. How to find one in your area.",
         "www.trustedchoice.com/independent-vs-direct-insurance"),
    ],

    # ── Scenario 6: Italian Restaurant ───────────────────────────────────
    "italian restaurant near me": [
        ("Best Italian Restaurants Near You | Yelp",
         "127 results sorted by rating and distance. Top rated: Trattoria Milano (4.7★, "
         "$$), Nonna's Kitchen (4.6★, $$$), Olive & Vine (4.5★, $$). Filter by price/distance.",
         "www.yelp.com/search?find_desc=italian+restaurant"),
        ("15 Best Italian Restaurants in Your Area | TripAdvisor",
         "Traveler-ranked Italian dining. #1 La Bella Vita: 'Authentic pasta, incredible "
         "tiramisu'. #2 Casa di Pasta: 'Best carbonara outside Rome'. Reservation links.",
         "www.tripadvisor.com/Restaurants-Italian-near-me"),
        ("OpenTable: Book Italian Restaurants Tonight",
         "23 Italian restaurants with availability tonight. Filter by time, party size, "
         "neighborhood, and price. Earn dining points. Most popular: 7:00-8:00 PM slots.",
         "www.opentable.com/italian-restaurants-near-me"),
        ("La Trattoria Authentic Italian - Menu & Hours",
         "Family-owned since 1987. Handmade pasta daily. Wood-fired pizza. Wine list with "
         "60+ Italian selections. Private dining room available. Open Tue-Sun 5-10 PM.",
         "www.latrattoria.com"),
        ("Ristorante Paradiso - Fine Italian Dining",
         "Chef Marco's tasting menu $85/person. House-made ravioli, osso buco, and "
         "signature limoncello cake. Romantic atmosphere. Reservations recommended.",
         "www.ristoranteparadiso.com"),
        ("Google Maps: Italian Restaurants Nearby",
         "Map showing 34 Italian restaurants within 5 miles. Average rating: 4.2★. Price "
         "range: $-$$$$. Popular times, photos, and directions for each location.",
         "www.google.com/maps/search/italian+restaurant"),
        ("Best Italian Restaurants for Birthday Dinners | Eater",
         "Curated list of Italian restaurants perfect for celebrations. Private rooms, "
         "special menus, and complimentary dessert options. Booking tips included.",
         "www.eater.com/italian-restaurants-birthday-dinner"),
        ("Pizza vs Pasta: Top Italian Restaurants Ranked by Specialty",
         "Best for pizza: Napoli Express, Slice of Italy. Best for pasta: Nonna's Kitchen, "
         "Pasta Fresca. Best for seafood: Mare Blu. Best for steaks: Bistecca House.",
         "www.thrillist.com/eat/best-italian-restaurants-by-specialty"),
        ("r/foodie - Underrated Italian restaurant recommendations?",
         "Local community shares hidden gems. Top responses mention family-run trattorias "
         "with no website, BYOB Italian spots, and new openings this month.",
         "www.reddit.com/r/foodie/underrated_italian_restaurants"),
        ("Italian Restaurant Etiquette: What to Know Before You Go",
         "Tipping guidelines, dress codes, how to order wine, and Italian dining customs. "
         "When to arrive, how long dinner typically takes, and menu terms explained.",
         "www.foodandwine.com/italian-dining-etiquette-guide"),
    ],

    # ── Scenario 7: Laptop for Video Editing ─────────────────────────────
    "laptop for video editing under 1500": [
        ("Best Laptops for Video Editing Under $1500 (2024) | Tom's Guide",
         "Top picks: MacBook Air M3 ($1,299), ASUS ProArt Studiobook ($1,399), Lenovo "
         "Legion Pro 5 ($1,449). Tested with DaVinci Resolve, Premiere Pro, and Final Cut.",
         "www.tomsguide.com/best-picks/best-laptops-for-video-editing-under-1500"),
        ("MacBook Air M3 for Video Editing: Can It Handle 4K?",
         "16GB unified memory handles 4K ProRes smoothly. 10-core GPU renders 20% faster "
         "than M2. Battery lasts 14 hours. Limitation: single external display support.",
         "www.macworld.com/article/macbook-air-m3-video-editing-review"),
        ("ASUS ProArt Studiobook 16 OLED Review: Creator's Dream",
         "Ryzen 9 + RTX 4060 8GB. 3.2K OLED display with 100% DCI-P3. 32GB RAM, 1TB SSD. "
         "ASUS Dial for creative control. Excellent color accuracy out-of-box. $1,399.",
         "www.notebookcheck.net/ASUS-ProArt-Studiobook-16-OLED-Review.html"),
        ("Best GPU for 4K Video Editing in Laptops: RTX 4060 vs M3 vs RX 7600S",
         "Benchmark comparison across Premiere Pro, DaVinci Resolve, and After Effects. "
         "RTX 4060 wins in Resolve, M3 wins in Final Cut, RX 7600S close behind.",
         "www.pugetsystems.com/labs/laptop-gpu-video-editing-benchmark"),
        ("How Much RAM Do You Actually Need for Video Editing?",
         "16GB: adequate for 1080p, tight for 4K. 32GB: comfortable for 4K, handles "
         "multicam. 64GB: only for 8K or heavy After Effects. Real-world RAM usage measured.",
         "www.premiumbeat.com/blog/how-much-ram-video-editing"),
        ("Lenovo Legion Pro 5 16IRX9: Budget 4K Editing Beast",
         "Intel i7-14700HX + RTX 4060. 16-inch 165Hz display (covers 99% sRGB). 32GB DDR5, "
         "1TB SSD. Excellent thermals under sustained rendering. Heavy at 2.5kg. $1,449.",
         "www.laptopmag.com/reviews/lenovo-legion-pro-5-16irx9"),
        ("Dell XPS 16 (2024) Review: Premium Editing on the Go",
         "Intel Core Ultra 9 + RTX 4070. Stunning 4K+ OLED touch display. 32GB RAM. "
         "Slim form factor at 2.0kg. Starts at $1,499 — right at budget limit.",
         "www.theverge.com/reviews/dell-xps-16-2024"),
        ("Video Editing Laptop Buying Guide: What Specs Matter Most",
         "Priority order: RAM ≥ 16GB > GPU with VRAM ≥ 6GB > SSD speed > CPU cores > "
         "display color accuracy. Why integrated graphics won't cut it for 4K timelines.",
         "www.creativebloq.com/buying-guides/video-editing-laptop"),
        ("r/VideoEditing - Best laptop under $1500 for Premiere Pro?",
         "Community recommendations: MacBook Air M3 for portability, Legion Pro 5 for raw "
         "power, ProArt for color accuracy. Discussion on SSD storage needs (1TB minimum).",
         "www.reddit.com/r/VideoEditing/laptop_under_1500_premiere"),
        ("External SSD vs Internal: Optimizing Storage for Video Projects",
         "Thunderbolt 4 external SSDs hit 2800 MB/s — fast enough for 4K proxy editing. "
         "Workflow: edit from external, render to internal. Best budget SSDs reviewed.",
         "www.bhphotovideo.com/explora/video/tips/external-ssd-video-editing"),
    ],

    # ── Scenario 8: Vitamin D Deficiency ─────────────────────────────────
    "symptoms of vitamin d deficiency": [
        ("Vitamin D Deficiency: Symptoms, Causes, and Treatment | Mayo Clinic",
         "Common symptoms: fatigue, bone pain, muscle weakness, depression. Risk factors: "
         "limited sun exposure, darker skin, age over 65, obesity. Blood test recommended.",
         "www.mayoclinic.org/diseases-conditions/vitamin-d-deficiency"),
        ("12 Signs and Symptoms of Vitamin D Deficiency | Healthline",
         "1. Fatigue 2. Bone pain 3. Muscle weakness 4. Mood changes 5. Frequent illness "
         "6. Slow wound healing 7. Hair loss 8. Back pain. When to see your doctor.",
         "www.healthline.com/nutrition/vitamin-d-deficiency-symptoms"),
        ("How Much Vitamin D Do You Need? RDA and Blood Level Guide",
         "RDA: 600-800 IU/day for adults. Optimal blood level: 30-50 ng/mL. Many experts "
         "recommend 1000-4000 IU/day for deficient individuals. D3 vs D2 explained.",
         "www.webmd.com/vitamins-and-supplements/vitamin-d-daily-requirements"),
        ("Vitamin D Deficiency Test: What to Expect and How to Read Results",
         "25-hydroxyvitamin D blood test is the gold standard. Deficient: <20 ng/mL, "
         "Insufficient: 20-29 ng/mL, Sufficient: 30-100 ng/mL. Costs $20-50.",
         "www.verywellhealth.com/vitamin-d-blood-test"),
        ("Can Vitamin D Deficiency Cause Fatigue? The Science Explained",
         "Meta-analysis of 28 studies: vitamin D deficiency associated with 2.3x higher "
         "odds of fatigue. Supplementation showed significant improvement in 8-12 weeks.",
         "www.ncbi.nlm.nih.gov/vitamin-d-fatigue-meta-analysis"),
        ("Best Vitamin D Supplements 2024: Third-Party Tested",
         "Top picks: Nature Made D3 2000 IU, NOW Foods D3 5000 IU, Thorne Liquid D3. "
         "All USP-verified. Take with fatty meals for 50% better absorption.",
         "www.consumerlab.com/reviews/vitamin-d-supplements"),
        ("Vitamin D Deficiency: Who Is Most at Risk?",
         "High-risk groups: office workers, elderly, people with dark skin, vegans, obese "
         "individuals, those with malabsorption conditions. Geographic latitude matters.",
         "www.clevelandclinic.org/vitamin-d-deficiency-risk-factors"),
        ("Vitamin D and Depression: What the Research Shows",
         "Systematic review: low vitamin D levels correlate with 14% increased depression risk. "
         "Supplementation may help, but is not a replacement for treatment. Talk to doctor.",
         "www.psychologytoday.com/vitamin-d-depression-research"),
        ("r/AskDocs - Could my tiredness be vitamin D deficiency?",
         "Doctor responses: fatigue has many causes. Vitamin D deficiency is common but not "
         "the only explanation. Recommended: blood work including CBC, thyroid, B12, iron, D.",
         "www.reddit.com/r/AskDocs/tiredness_vitamin_d"),
        ("Sun Exposure vs Supplements: Getting Enough Vitamin D Naturally",
         "10-30 minutes midday sun 2-3x/week may suffice (varies by skin tone and latitude). "
         "Winter months at >37°N latitude: supplementation usually necessary.",
         "www.health.harvard.edu/vitamin-d-sun-vs-supplements"),
    ],

    # ── Scenario 9: Credit Card Cashback ─────────────────────────────────
    "best credit card cashback rewards": [
        ("Best Cashback Credit Cards of 2024 | NerdWallet",
         "Top picks: Citi Double Cash (2% flat), Chase Freedom Unlimited (1.5% + 5% rotating), "
         "Blue Cash Preferred (6% groceries). No annual fee vs premium options compared.",
         "www.nerdwallet.com/best/credit-cards/cash-back"),
        ("Wells Fargo Active Cash® Card: 2% Cash Back on Everything",
         "Flat 2% cash rewards on all purchases. $200 welcome bonus after $500 spend. "
         "No annual fee. 0% intro APR for 12 months. Cell phone protection included.",
         "www.wellsfargo.com/credit-cards/active-cash"),
        ("Chase Freedom Unlimited vs Citi Double Cash: Which Is Better?",
         "CFU: 1.5% base + 5% travel + 3% dining/drugstores. Double Cash: flat 2% on "
         "everything. For groceries-heavy spenders, Double Cash wins. For varied spending, CFU.",
         "www.creditcards.com/compare/freedom-unlimited-vs-double-cash"),
        ("Best Credit Cards for Groceries: Up to 6% Cash Back",
         "Blue Cash Preferred: 6% groceries (up to $6k/year), $95 annual fee. Blue Cash "
         "Everyday: 3% groceries, no fee. Capital One SavorOne: 3% + dining + streaming.",
         "www.bankrate.com/credit-cards/best-grocery-rewards"),
        ("How Credit Card Cashback Actually Works (And the Math Behind It)",
         "Statement credits vs direct deposit vs points conversion. Flat-rate vs category vs "
         "rotating categories. Calculate: $2k/month spending × 2% = $480/year cashback.",
         "www.investopedia.com/how-cash-back-credit-cards-work"),
        ("r/creditcards - Best cashback setup for maximizing rewards?",
         "Community optimal setup: Citi Custom Cash (5% top category), Amex BCP (6% grocery), "
         "Chase Freedom Flex (5% rotating), Citi Double Cash (2% everything else).",
         "www.reddit.com/r/creditcards/best_cashback_setup_maximize"),
        ("Capital One SavorOne: Best No-Annual-Fee Dining Card",
         "3% on dining, entertainment, groceries, and streaming. 1% on everything else. "
         "$200 bonus. No foreign transaction fees. Unlimited cashback, never expires.",
         "www.capitalone.com/credit-cards/savorone-cash-rewards"),
        ("Credit Card Cashback vs Travel Points: Which Rewards Are Worth More?",
         "Cashback: simple, 1-2% value. Travel points: complex, 1.5-3% value if optimized. "
         "For casual users, cashback wins. For frequent travelers, points can be 2x better.",
         "www.thepointsguy.com/guide/cash-back-vs-travel-points"),
        ("Best Credit Cards with No Annual Fee (2024 Edition)",
         "Citi Double Cash, Chase Freedom Unlimited, Discover it Cash Back. All offer "
         "competitive rewards without paying an annual fee. Compare sign-up bonuses.",
         "www.usnews.com/credit-cards/no-annual-fee"),
        ("How to Maximize Your Cashback: Multi-Card Strategy Guide",
         "Use 3-4 cards strategically. Assign each to its highest-reward category. "
         "Automate payments. Track spending by category. Average optimized return: 3.2%.",
         "www.doctorofcredit.com/maximize-cashback-multi-card-strategy"),
    ],

    # ── Scenario 10: Online MBA Programs ─────────────────────────────────
    "online mba programs ranking": [
        ("Best Online MBA Programs 2024 | US News Rankings",
         "#1 Indiana University (Kelley), #2 Carnegie Mellon (Tepper), #3 UNC (Kenan-Flagler). "
         "Rankings based on peer assessment, student engagement, admissions selectivity.",
         "www.usnews.com/education/online-education/mba/rankings"),
        ("Top 25 Most Affordable Online MBA Programs Under $25,000",
         "University of the People ($2,460 total), Fort Hays State ($14,436), "
         "University of Arkansas ($23,480). AACSB-accredited programs prioritized.",
         "www.affordablecollegesonline.org/degrees/mba-programs"),
        ("Is an Online MBA Worth It? ROI Analysis with Real Data",
         "Average salary increase: 33% within 3 years. Top programs: 77% report promotion. "
         "Break-even analysis: $30K program pays for itself in 18 months at $15K raise.",
         "www.mba.com/explore-programs/online-mba-roi-analysis"),
        ("Indiana University Kelley Online MBA: In-Depth Program Review",
         "2-year or 3-year pace. Total cost: $74,520. Live online classes with Kelley Direct "
         "cohort. Strong alumni network. Graduate with same degree as on-campus students.",
         "www.poets-and-quants.com/review/kelley-online-mba"),
        ("AACSB vs AMBA vs EQUIS: Why MBA Accreditation Matters",
         "Only 5% of business schools hold AACSB accreditation. Triple-crown (all three) "
         "held by 1%. Why employers care and how to verify accreditation status.",
         "www.mba.com/accreditation-guide"),
        ("Online MBA Admission Requirements: What You Need to Apply",
         "Typical: bachelor's degree, 2+ years work experience, GMAT/GRE (many now optional), "
         "essays, letters of recommendation, resume. Application timeline guide.",
         "www.princetonreview.com/mba/online-mba-admissions"),
        ("Part-Time vs Full-Time vs Online MBA: Which Format Is Right?",
         "Online: most flexible, work full-time. Part-time: some in-person networking. "
         "Full-time: fastest, best for career changers. Cost comparison and time commitment.",
         "www.businessbecause.com/mba/part-time-vs-full-time-vs-online"),
        ("r/MBA - Online MBA from top-20 school vs in-person at unranked?",
         "Consensus: top-20 online MBA beats unranked in-person for networking and brand. "
         "Exception: if local school has strong regional employer connections.",
         "www.reddit.com/r/MBA/online_top20_vs_inperson_unranked"),
        ("How to Choose an Online MBA Program: 8 Key Factors",
         "1. Accreditation 2. Rankings 3. Specialization options 4. Cost/financial aid "
         "5. Class format (sync vs async) 6. Residency requirements 7. Alumni network 8. ROI.",
         "www.fortune.com/education/articles/choose-online-mba-program"),
        ("Wharton vs Chicago Booth vs Columbia: Elite Online MBAs Compared",
         "All three now offer online/hybrid options. Cost: $160-220K. Selectivity: 20-30% "
         "acceptance. Curriculum differences, global immersion trips, and capstone projects.",
         "www.efinancialcareers.com/wharton-booth-columbia-online-mba"),
    ],

    # ── Default templates for reformulated or unseen queries ─────────────
    "default": [
        ("Comprehensive Guide: {query}",
         "Everything you need to know about {query}. Expert analysis, comparisons, and "
         "step-by-step recommendations based on the latest data and user reviews.",
         "www.expertguide.com/{slug}"),
        ("{query} - Top 10 Results Reviewed and Compared",
         "We researched and compared the top options for {query}. Rankings based on "
         "quality, value, user satisfaction, and expert assessments. Updated monthly.",
         "www.toptenreviews.com/{slug}"),
        ("Best {query} in 2024: Expert Picks",
         "Updated rankings and reviews for {query}. Our team tested and evaluated each "
         "option for performance, reliability, and overall value.",
         "www.bestof2024.com/{slug}"),
        ("Reddit Discussion: {query}",
         "Community opinions, real experiences, and honest recommendations related to "
         "{query}. Upvoted answers from verified users and domain experts.",
         "www.reddit.com/search?q={slug}"),
        ("{query} - Wikipedia",
         "Overview, background information, history, and key concepts related to {query}. "
         "References and external links for deeper research.",
         "en.wikipedia.org/wiki/{slug}"),
        ("How to Evaluate {query}: A Buyer's Guide",
         "Step-by-step framework for making the best decision regarding {query}. "
         "Key factors to consider, common mistakes, and expert tips.",
         "www.consumersadvocate.org/{slug}"),
        ("{query}: Complete Comparison Chart",
         "Side-by-side comparison of all major options for {query}. Price, features, "
         "pros/cons, and user ratings in an interactive table.",
         "www.versus.com/compare/{slug}"),
        ("Expert Tips and Advice: {query}",
         "Industry professionals and experienced users share their best advice on "
         "{query}. Practical tips, cost-saving strategies, and pitfalls to avoid.",
         "www.expertadvice.com/tips/{slug}"),
        ("{query} - Latest News and Updates (2024)",
         "Recent developments, product launches, policy changes, and trending topics "
         "related to {query}. Updated daily from trusted sources.",
         "www.news.com/topic/{slug}"),
        ("Frequently Asked Questions About {query}",
         "Answers to the 20 most common questions about {query}. Compiled from expert "
         "interviews, research papers, and thousands of user queries.",
         "www.faqhub.com/{slug}"),
        ("{query} for Beginners: Where to Start",
         "New to {query}? This beginner's guide covers the fundamentals, common "
         "terminology, and a clear roadmap to get started today.",
         "www.beginnerguide.com/{slug}"),
        ("{query}: What Experts Recommend in 2024",
         "Curated recommendations from industry experts and thought leaders on {query}. "
         "Includes budget options, premium choices, and best overall picks.",
         "www.expertpicks.com/2024/{slug}"),
    ],
}


AD_TEMPLATES: Dict[str, List[ResultTemplate]] = {

    "best noise cancelling headphones": [
        ("Shop Noise Cancelling Headphones - Up to 40% Off | Amazon",
         "Deals on Sony, Bose, Apple, Sennheiser. Free 2-day shipping with Prime. "
         "30-day return policy. Customer ratings and verified purchase reviews.",
         "www.amazon.com/ad/noise-cancelling-headphones"),
        ("Bose Official Store - QuietComfort Ultra $349 + Free Shipping",
         "Buy direct from Bose. 90-day risk-free trial. Free returns. "
         "Exclusive color options not available at retailers.",
         "www.bose.com/ad/quietcomfort-ultra-shop"),
        ("Best Buy: Headphone Deals This Week - Price Match Guarantee",
         "Save $50-100 on select noise cancelling headphones. Same-day delivery or "
         "free in-store pickup. Expert setup assistance included.",
         "www.bestbuy.com/ad/headphone-deals"),
        ("Sony WH-1000XM5 Refurbished - Certified Like New $228",
         "Factory-refurbished with full warranty. Save $121 vs new. Ships same day. "
         "90-day satisfaction guarantee or money back.",
         "www.secondipity.com/ad/sony-xm5-refurbished"),
    ],

    "cheap flights to tokyo": [
        ("Flights to Tokyo from $398 Round-Trip | Priceline",
         "Exclusive deals on Tokyo flights. Name Your Own Price® for extra savings. "
         "Price match guarantee. Flexible cancellation options available.",
         "www.priceline.com/ad/flights-tokyo"),
        ("Japan Airlines - Premium Economy to Tokyo from $799",
         "Extra legroom, priority boarding, and premium meal service. Book direct "
         "and earn double Mileage Bank miles. Flexible fare available.",
         "www.jal.com/ad/premium-economy-deals"),
        ("Expedia Tokyo Flight + Hotel Packages: Save Up to $200",
         "Bundle and save on Tokyo vacation packages. 4-star hotels from $89/night "
         "when booked with flights. Free cancellation on select packages.",
         "www.expedia.com/ad/tokyo-flight-hotel-deals"),
    ],

    "python machine learning tutorial": [
        ("DataCamp: Learn Machine Learning with Python - First Month Free",
         "Interactive coding exercises with real datasets. 80+ ML courses. "
         "Certificate of completion. Used by 13M learners worldwide.",
         "www.datacamp.com/ad/machine-learning-python"),
        ("Coursera: Machine Learning Specialization by Andrew Ng",
         "Stanford University course. Financial aid available. Earn a certificate "
         "in 3 months. 4.9★ rating from 180K+ learners.",
         "www.coursera.org/ad/machine-learning-specialization"),
        ("Udemy: Complete ML Bootcamp 2024 - $12.99 (92% Off)",
         "36-hour comprehensive course. 500+ coding exercises. Lifetime access. "
         "Projects: fraud detection, recommendation engine, NLP chatbot.",
         "www.udemy.com/ad/machine-learning-bootcamp"),
    ],

    "best running shoes 2024": [
        ("Nike.com: Running Shoes Starting at $80 - Free Shipping",
         "Shop Pegasus 41, Vomero 17, and Vaporfly 3. Nike Members get 20% off "
         "first order. 60-day wear test — return if not satisfied.",
         "www.nike.com/ad/running-shoes"),
        ("Running Warehouse: Expert Shoe Finder + Free Returns",
         "Answer 5 questions, get personalized shoe recommendations. Free 90-day "
         "returns on all shoes. Price match guarantee.",
         "www.runningwarehouse.com/ad/shoe-finder"),
        ("ASICS Official: Gel-Nimbus 26 - Shop Now",
         "FF BLAST PLUS ECO cushioning. Engineered mesh upper. Men's and women's "
         "available. $160. Free shipping on orders over $50.",
         "www.asics.com/ad/gel-nimbus-26"),
    ],

    "home insurance quotes comparison": [
        ("Progressive: Home Insurance Quote in 7 Minutes",
         "Average savings of $687/year when you switch. Bundle with auto for extra "
         "discount. A+ rated. Online claims filing. Get your personalized quote now.",
         "www.progressive.com/ad/homeowners-insurance"),
        ("Lemonade: Home Insurance from $25/month - Instant Quote",
         "AI-powered claims process. No brokers, no paperwork. Rated A by AM Best. "
         "Everything you need in an app. Cancel anytime.",
         "www.lemonade.com/ad/homeowners-insurance"),
        ("State Farm: Trusted Home Insurance Since 1922",
         "#1 home insurer in America. Local agent support. Discount for smart home "
         "devices. Bundle with auto and save up to 25%.",
         "www.statefarm.com/ad/homeowners-insurance"),
        ("Geico: Save on Home Insurance - Free Quote in Minutes",
         "Compare rates from top carriers through Geico. Average savings of $500/year. "
         "24/7 customer support. A++ financial rating.",
         "www.geico.com/ad/homeowners-insurance"),
    ],

    "italian restaurant near me": [
        ("Olive Garden: Order Online or Reserve a Table",
         "Never-ending pasta bowl $13.99. New seasonal menu. Earn reward points. "
         "Curbside pickup available. Find your nearest location.",
         "www.olivegarden.com/ad/reserve-now"),
        ("DoorDash: Italian Food Delivered in 30 Minutes",
         "$0 delivery fee on first 3 orders. Top-rated Italian restaurants in your "
         "area. Real-time tracking. Schedule ahead for tonight.",
         "www.doordash.com/ad/italian-restaurants"),
        ("OpenTable: Book Italian Tonight - Earn 1000 Bonus Points",
         "Limited-time promotion: 1000 bonus dining points when you book Italian. "
         "Available tables tonight starting at 6:00 PM.",
         "www.opentable.com/ad/italian-dining-bonus"),
    ],

    "laptop for video editing under 1500": [
        ("B&H Photo: Video Editing Laptops - Free Expedited Shipping",
         "Save on MacBook Air M3, ASUS ProArt, Lenovo Legion. Tax-free in most "
         "states. Expert advice available. 30-day return policy.",
         "www.bhphotovideo.com/ad/video-editing-laptops"),
        ("Apple Education Store: MacBook Air M3 from $1,199",
         "Students and educators save. 16GB RAM, 512GB SSD. Apple Care+ available. "
         "Free engraving. Trade in your old Mac for credit.",
         "www.apple.com/ad/education-macbook-air"),
        ("Best Buy: Creator Laptops on Sale This Week",
         "Up to $200 off select video editing laptops. Geek Squad setup included. "
         "Student discount available. Finance for $0 down, 0% APR.",
         "www.bestbuy.com/ad/creator-laptops-sale"),
    ],

    "symptoms of vitamin d deficiency": [
        ("Everlywell: At-Home Vitamin D Test - Results in Days",
         "FDA-cleared home collection kit. Easy finger-prick blood test. Digital "
         "results with physician review. $49. HSA/FSA eligible.",
         "www.everlywell.com/ad/vitamin-d-test"),
        ("Nature Made Vitamin D3: #1 Pharmacist Recommended Brand",
         "USP verified for purity and potency. 2000 IU softgels, 250 count. "
         "$12.99 at your local pharmacy. Buy online with Subscribe & Save.",
         "www.naturemade.com/ad/vitamin-d3"),
        ("Nurish by Nature Made: Personalized Vitamin Packs",
         "Take the quiz and get a customized daily vitamin pack. Includes D3, B12, "
         "and more based on your health goals. 30-day supply from $20.",
         "www.nurish.com/ad/vitamin-quiz"),
    ],

    "best credit card cashback rewards": [
        ("Chase Freedom Unlimited® - Earn $200 Bonus + 1.5% Cashback",
         "Unlimited 1.5% cash back. $200 bonus after $500 spend in first 3 months. "
         "No annual fee. 0% intro APR for 15 months.",
         "www.chase.com/ad/freedom-unlimited-cashback"),
        ("Citi® Double Cash Card - Earn 2% on Every Purchase",
         "1% when you buy, 1% when you pay. No categories to track. No annual fee. "
         "Balance transfer offer: 0% intro APR for 18 months.",
         "www.citi.com/ad/double-cash-card"),
        ("American Express Blue Cash Preferred® - 6% at Supermarkets",
         "6% cash back at US supermarkets (up to $6,000/year). 6% on streaming. "
         "3% at gas stations and transit. $95 annual fee. $350 welcome bonus.",
         "www.americanexpress.com/ad/blue-cash-preferred"),
        ("Capital One Quicksilver: Unlimited 1.5% Cash Back",
         "Simple flat-rate rewards. $200 cash bonus. No annual fee. No foreign "
         "transaction fees. Pre-qualify without affecting credit score.",
         "www.capitalone.com/ad/quicksilver-cash-rewards"),
    ],

    "online mba programs ranking": [
        ("Indiana University Kelley Online MBA - Apply Now",
         "#1 ranked online MBA (US News). GMAT waiver available. Flexible 2 or 3 "
         "year pace. $74,520 total. Same Kelley degree as on-campus. Request info.",
         "www.kelley.iu.edu/ad/online-mba-apply"),
        ("GMAT Prep by Magoosh: Score 700+ Guaranteed or Money Back",
         "3-month study plan. 800+ practice questions. Video lessons by top instructors. "
         "$219 — fraction of the cost of competitors. Mobile app included.",
         "www.magoosh.com/ad/gmat-prep"),
        ("UNC Kenan-Flagler Online MBA - Top 3 Ranked",
         "Global immersions, live classes, and a powerful alumni network. GMAT/GRE "
         "optional. Application deadline: March 15. Schedule a virtual info session.",
         "www.kenan-flagler.unc.edu/ad/online-mba"),
    ],

    "default": [
        ("Buy {query} - Best Deals Online | Top Retailers",
         "Compare prices from 50+ trusted retailers. Up to 50% off. Free shipping "
         "available. Customer reviews and ratings. Price alerts.",
         "www.shopping.com/ad/{slug}"),
        ("Official {query} Store - Direct from Manufacturer",
         "Best prices guaranteed when you buy direct. Free shipping and returns. "
         "Extended warranty options. Exclusive online-only deals.",
         "www.officialstore.com/ad/{slug}"),
        ("{query} - Compare Prices & Save | PriceGrabber",
         "Find the lowest prices from verified sellers. Price history charts. "
         "Coupon codes automatically applied. Save money today.",
         "www.pricegrabber.com/ad/{slug}"),
        ("Shop {query} at Target - Free Same-Day Delivery",
         "Free same-day delivery with Shipt on orders over $35. RedCard members "
         "save extra 5%. Easy returns in-store or by mail.",
         "www.target.com/ad/{slug}"),
    ],
}
