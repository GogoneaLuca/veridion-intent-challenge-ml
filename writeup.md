# Veridion Search Intent Challenge: Arhitectură Hibridă RAG
**Candidat:** Gogonea Luca

## Introducere: Filosofia de Bază a Soluției & Exploratory Data Analysis

Când am analizat prima dată cerințele acestui motor de căutare, punctul principal de fricțiune (bottleneck-ul) în căutările B2B a devenit evident: **Intenția umană este prin natura ei ambiguă (fuzzy), dar bazele de date necesită o logică matematică strictă.** Dacă un utilizator caută *"Fast-growing fintechs competing with traditional banks"*, un query SQL standard eșuează, deoarece "competing with banks" este un concept abstract. Invers, dacă ne bazăm exclusiv pe căutarea vectorială (Baseline B), matematica eșuează pe constrângerile numerice stricte — modelele de *embeddings* nu pot distinge cu precizie diferența critică dintre "revenue > $50M" și "revenue < $50M", deoarece distanța semantică dintre cele două fraze este aproape identică. Pe de altă parte, o evaluare pură cu un LLM (Baseline A) a tuturor celor 500 de companii face sistemul inacceptabil de lent și costisitor.

Înainte de a proiecta arhitectura care să rezolve aceste limitări, am luat două decizii fundamentale. 

Prima a fost studierea atentă a bazei de date furnizate prin crearea unui script de **Exploratory Data Analysis (EDA)** (vezi `data_exploration.py` atașat în repository). Analiza a scos la iveală lipsuri critice (un procent semnificativ de valori `NaN` la numărul de angajați și anul înființării) și neuniformități majore (date geografice stocate ca string-uri de dicționare). Acest profilaj mi-a demonstrat că sistemul nu se poate baza pe filtre rigide și necesită un grad mare de toleranță la date incomplete.

A doua decizie a fost o documentare aprofundată a filosofiei tehnice Veridion, urmărind playlist-ul de YouTube **"How Veridion Works"**. De acolo am extras conceptul central al arhitecturii mele: **abordarea în "Pâlnie" (The Funnel Approach).** Așa cum Veridion procesează miliarde de pagini web nestructurate, trecându-le prin filtre de complexitate crescătoare, am decis să construiesc un pipeline care filtrează brutal datele ieftine matematic la început, lăsând analiza semantică grea și nuanțată (LLM) doar pentru final.

**Întrebarea care a definit această arhitectură a fost:** *"Cum pot decupla constrângerile matematice stricte (numere, locații) de intenția semantică abstractă (industrii, concepte), procesându-le prin această 'pâlnie' pentru a optimiza simultan acuratețea și latența?"*

Rezultatul este un **Pipeline Hibrid Retrieve-and-Rank în 3 Etape**, construit pentru a elimina agresiv nepotrivirile evidente în prima fază, lăsând doar un subset extrem de relevant pentru o analiză semantică profundă, dar costisitoare computațional.

## 1. Arhitectura Sistemului (The Pipeline)

Sistemul este segmentat în trei module independente. Fiecare componentă este optimizată pentru un task specific, echilibrând viteza, precizia și costul de inferență.

### Modulul 1: The Intent Parser (Creierul)
* **Mecanism:** Utilizează modelul `llama-3.3-70b` (rulat prin infrastructura ultra-rapidă LPU de la Groq), configurat strict cu `response_format: json_object` și o schemă Pydantic.
* **Rolul:** Funcționează ca un motor de extragere determinist. În loc să trateze query-ul ca pe un simplu bloc de text, Parser-ul îl descompune în două output-uri distincte:
    1. **Hard Filters:** Limite numerice exacte (angajați, an, venituri) și constrângeri geografice. Pentru a asigura compatibilitatea perfectă cu baza de date, LLM-ul este instruit să traducă dinamic conceptele regionale (ex: "Europe" sau "Scandinavia") direct în liste de coduri de țară ISO-2.
    2. **Semantic Intent:** Un string scurt și curățat (ex: "clean energy startup"), complet eliberat de cifrele și locațiile care ar "polua" modelul vectorial.

### Modulul 2: The Fast Retriever (Mușchiul și Logica Matematică)
* **Mecanism:** Un motor hibrid care rulează 100% local, offline, combinând eficiența filtrării tabulare (Pandas) cu puterea căutării semantice (modelul `sentence-transformers/all-MiniLM-L6-v2`). Am ales rularea modelului de *embeddings* local pentru a elimina latența de rețea și dependența de API-uri terțe la trecerea prin mii de înregistrări.
* **Rolul și Raționamentul:** Pentru a menține timpul de răspuns sub câteva secunde, acest modul nu citește orbește toate datele, ci execută o "pâlnie" (funnel) în doi pași foarte stricți:

    **1. Pre-filtrarea Tabulară (Pandas) și Dilema Datelor Lipsă (`NaN`):**
    Primul pas taie instantaneu spațiul de căutare aplicând *Hard Filters* extrase de Modulul 1. Aici m-am confruntat cu o problemă clasică din lumea reală (descoperită în timpul analizei exploratorii - EDA): multe companii și startup-uri relevante aveau valori lipsă (`NaN`) la anul înființării sau numărul de angajați. 
    * *Decizia de Design:* Dacă aș fi aplicat o filtrare rigidă (ex: `angajați < 200`), sistemul ar fi eliminat automat aceste companii promițătoare. Pentru a preveni acest lucru, am implementat o logică de **Soft-Fail**. Codul permite trecerea mai departe a companiilor care fie respectă exact condiția, fie au datele lipsă. 
    * *Trade-off:* Prin această decizie am ales să maximizez **Recall-ul** (să nu pierd candidați buni) în detrimentul unei **Precizii** stricte pe moment. Am acceptat riscul de a aduce puțin "zgomot" în faza următoare, bazându-mă pe inteligența Modulului 3 (Evaluatorul Final) pentru a curăța rezultatele false la final.

    **2. Vector Search și Demistificarea Potrivirii Semantice:**
    Odată ce Pandas a eliminat companiile imposibile matematic sau geografic, intervine motorul de Machine Learning. În loc să caut potriviri exacte de cuvinte (*keyword matching*), care ratează sinonimele, modelul transformă intenția utilizatorului și descrierea fiecărei companii în vectori densi (liste de numere plasate într-un spațiu matematic multidimensional).
    * *Cum calculez similaritatea:* Pentru a găsi cele mai bune potriviri, am folosit metrica **Cosine Similarity**. În esență, acest algoritm măsoară unghiul dintre vectorul căutării și vectorul companiei. Dacă unghiul este foarte mic (textele "țintesc" în aceeași direcție semantică), scorul se apropie de 1. Acest lucru îi permite sistemului meu să înțeleagă că o firmă descrisă ca oferind "banking alternatives" se potrivește perfect cu o căutare pentru "fintech", chiar dacă nu împart niciun cuvânt comun. Această etapă sortează candidații rămași și trimite doar Top 10 către faza finală.

### Modulul 3: The Deep Judge (LLM-as-a-Judge Reranker)
* **Mecanism:** Trimite cele mai bune 10 companii returnate de Modulul 2 înapoi către LLM, într-un singur prompt de tip *batch*.
* **Rolul:** Acționează ca un evaluator final, capabil să înțeleagă nuanțe. Deoarece spațiul de căutare a fost redus masiv, ne permitem să folosim capacitatea avansată de raționament a unui LLM fără a distruge timpul de răspuns.
* **Engineering Polish:** Pentru a garanta un *scoring* determinist și granular, am implementat o metodologie **Chain-of-Thought (CoT)** dublată de o grilă de notare (Tiered Rubric). Schema forțează modelul să își genereze logica (`reasoning`) *înainte* de a acorda scorul final de relevanță (0-100). Aceasta previne colapsarea notelor în valori rotunde (ex: 80, 90) și oferă utilizatorului un model de *Explainable AI* — motivul exact pentru care o companie i-a fost recomandată.

## 2. Overcoming the Baselines Limitations
În analiza documentației inițiale, am identificat limitările critice ale celor două arhitecturi de bază propuse și am iterat pentru a le mitiga direct.

**Solving Baseline A (The Latency Bottleneck):**
Baseline-ul A propunea evaluarea individuală a companiilor de către un LLM, ducând la o latență inacceptabilă. Am rezolvat această problemă prin utilizarea infrastructurii **LPU (Language Processing Unit)** de la Groq și prin evaluarea în **Batch** (trimiterea tuturor celor 10 candidați într-un singur prompt). Această abordare a redus timpul de inferență pentru faza de Reranking la sub 2 secunde.

**Solving Baseline B (The "Noisy" Embeddings):**
Baseline-ul B propunea transformarea întregului query într-un singur vector. Am observat că modelele dense performează slab la "potrivirea exactă" a numerelor (ex: "> 1000 employees") generând False Positives. Sistemul meu deleagă numerele către Modulul 1 (ca filtre stricte) și trimite către Modulul 2 doar "esența" intenției, crescând dramatic precizia căutării.

## 3. Engineering Decisions, Reasoning & Trade-offs

Pe parcursul dezvoltării, arhitectura a suferit multiple iterații. În ingineria datelor, prima soluție este rareori cea optimă. Mai jos am documentat procesul meu de gândire, ipotezele testate și compromisurile (trade-offs) asumate conștient.

### 3.0. Exploratory Data Analysis (EDA) ca Fundație a Arhitecturii
Înainte de a proiecta orice modul al sistemului, am dezvoltat un script de analiză (`data_exploration.py`, atașat în repository) pentru a evalua structura, calitatea și rata de completare a dataset-ului de 500 de companii. 

În urma rulării acestui profilaj de date, am extras următoarele concluzii care au dictat designul sistemului:
* **Problema datelor lipsă:** Am descoperit un procent semnificativ de valori lipsă (`NaN`) în câmpuri esențiale pentru filtrare, precum `employee_count` și `year_founded`. Acest insight m-a forțat să abandonez o filtrare Pandas rigidă și să implementez abordarea de tip *Soft-Fail* în Modulul 2.
* **Neuniformitatea formatelor:** Datele geografice (`address`) și clasificările industriale erau stocate sub formă de string-uri de dicționare. Această lipsă de standardizare mi-a confirmat că parsarea tradițională ar fi prea fragilă, justificând decizia de a folosi capabilitățile de extracție ale LLM-ului (Modulul 1) pentru a mapa dinamic locațiile la standardul ISO-2.
* **Baza decizională:** Analiza a demonstrat că un sistem robust trebuie să compenseze "zgomotul" și lipsele din datele B2B reale.

### 3.1. What did I optimize for? (The Trade-offs)
Atunci când proiectezi un sistem de Search B2B hibrid, este imposibil să optimizezi pentru toate metricile simultan. 
* **Optimizarea principală:** Arhitectura mea a fost construită cu un focus obsesiv pe **Acuratețe (Accuracy)** și **Robustețe (Robustness)**. Am vrut un sistem care să nu fie păcălit de capcane semantice și care să poată extrage startup-uri ascunse din profile cu date incomplete.
* **Trade-off asumat:** Pentru a obține această acuratețe, am făcut un compromis conștient renunțând la **Simplitate (Simplicity)** arhitecturală și, într-o oarecare măsură, la **Cost/Viteză Extremă**. Am preferat un sistem mai complex (cu 3 etape distincte) și utilizarea unui LLM masiv (70B parametri via API) în etapa finală, considerând că un output de înaltă calitate (Reranking cu explicații logice) este mai valoros într-un produs B2B decât o căutare pur vectorială de 10 milisecunde, dar inexactă (Baseline B).

### 3.2. Dynamic Geo-Expansion vs. Hardcoded Dictionaries
* **Întrebarea pe care mi-am pus-o:** *Cum gestionez query-uri geografice macro ("Scandinavia", "Europe") când baza de date folosește strict coduri ISO-2 ("se", "fr", "ro")?*
* **Abordarea inițială:** Am construit un dicționar Python hardcodat pentru a mapa regiunile la coduri ISO. 
* **Failure Mode (Analiza Erorii):** La testarea interogării *"Pharmaceutical companies in Switzerland"*, sistemul a returnat zero rezultate. Făcând debugging, am descoperit că, neavând "Switzerland" în dicționar, codul a aplicat o regulă generică de tăiere a primelor două litere ("sw" în loc de "ch"), eșuând complet.
* **Decizia finală:** Am realizat că mentenanța unui dicționar global este un *bottleneck* masiv. Am eliminat complet dicționarul și am delegat componenta de Geo-Mapping către Modulul 1 (LLM). Prin prompt engineering, modelul returnează direct un array de coduri ISO-2. 
* **Trade-off:** Am tranzacționat determinismul codului pe flexibilitatea LLM-ului. Deși există un risc minor de halucinație la nivel de coduri ISO, am câștigat scalabilitate globală instantanee fără costuri de latență la filtrare.

### 3.3. The Missing Data Dilemma (Precision vs. Recall)
* **Întrebarea:** *Dacă un utilizator caută "startups under 200 employees", iar o companie perfect relevantă are valoarea `NaN` la coloana `employee_count`, o eliminăm?*
* **Decizia:** Am implementat o logică intenționată de **Soft-Fail** în filtrarea din Pandas (`mask = mask | df['employee_count'].isna()`). Companiile cu date lipsă sunt lăsate să treacă în etapa de Reranking semantic.
* **Trade-off:** Am optimizat explicit pentru **Recall** (maximizarea descoperirilor) în detrimentul **Preciziei** inițiale. Într-o bază de date B2B, profilele incomplete sunt la ordinea zilei. Am preferat să accept un volum mai mare de "zgomot" în Modulul 2, pe care l-am lăsat în sarcina Modulului 3 să îl filtreze deductiv, decât să pierd candidați ideali din cauza unui simplu câmp `null`.

### 3.4. Mitigating LLM Instability (Cost vs. Acuratețe Deterministică)
* **Întrebarea:** *Cum mă asigur că un LLM, prin natura lui probabilistic, acordă note consecvente aceleiași companii la rulări diferite?*
* **Observația:** În primele iterații, Modulul 3 (The Deep Judge) oferea scoruri puternic rotunjite (70, 80, 90) și genera justificări repetitive.
* **Soluția tehnică:** Am implementat o arhitectură **Chain-of-Thought (CoT)** în schema Pydantic. Am forțat modelul să scrie câmpul `reasoning` *înainte* de a emite `relevance_score`-ul. Adăugând o grilă de corectare pe paliere, am obținut un output de tip *Explainable AI* stabil, cu granularitate fină (ex: 94/100) și justificări logice, nu doar descriptive.

### 3.5. Error Analysis: Capcana "False Positives" (Studiu de caz: Shopify)
* **Întrebarea:** *Ce se întâmplă când sistemul este forțat să caute o condiție extrem de specifică ce lipsește cu desăvârșire din baza de date?*
* **Observația în Testare:** La evaluarea interogării *"E-commerce companies using Shopify"*, Modulul 2 (Vector Search) a regăsit giganți precum Walmart sau Coles, atrași puternic de vectorul "E-commerce". Totuși, Modulul 3 le-a depunctat masiv (scoruri < 40/100).
* **Concluzia:** Confirmând prin EDA că *nicio* companie din cele 500 nu folosește Shopify, faptul că Modulul 3 a sesizat absența acestui criteriu critic și a refuzat să acorde un scor mare validează arhitectura. Sistemul nu este un simplu "keyword matcher", ci posedă capacitatea de a judeca absența intenției.

### 3.6. Error Analysis: Where does the system struggle?
Niciun sistem nu este perfect. Analizând output-ul pe cele 12 interogări de test, am identificat unde sistemul are cele mai mari dificultăți: **"The Marketing Fluff Vulnerability"** (vulnerabilitatea la limbaj promoțional). Sistemul vectorial (Modulul 2) este uneori orbit de vocabularul specific unei industrii, ignorând modelul real de business.
* **Concrete Misclassification Example:** La rularea interogării *"Fast-growing fintech companies competing with traditional banks"*, Modulul 2 a adus în Top 10 candidați compania **Ford Credit**. 
* **De ce a fost clasificată greșit inițial?** Deoarece descrierea Ford Credit abundă în termeni financiari ("financing", "leasing", "credit", "insurance"). În spațiul latent vectorial (Cosine Similarity), acești termeni plasează compania foarte aproape de conceptul de "bancă" sau "fintech". Sistemul matematic este orb la "nuanța" de inovație tehnologică cerută de cuvântul *fintech*, clasificând greșit o divizie auto tradițională drept o companie de tehnologie financiară.
* **Cum este mitigată:** Din fericire, Modulul 3 a identificat anomalia și i-a acordat un scor de doar **42/100**, notând clar că *"oferă servicii financiare auto, dar nu este un fintech în sensul clasic"*. Totuși, faptul că a ajuns în primele 10 rezultate arată o limitare a *Semantic Intent-ului* .

### 3.7. Scaling, Failure Modes & Production Monitoring
Pentru a scala de la 500 la sute de mii (sau milioane) de companii, arhitectura actuală necesită modificări structurale.

**Scaling Bottlenecks (Ce aș schimba):**
1. **API Rate Limiting:** În timpul testelor, am epuizat limita zilnică de 100.000 de tokeni oferită de Groq. La scară largă, trimiterea de batch-uri către un LLM masiv devine un bottleneck. **Soluția:** Aș înlocui complet Modulul 3 cu un model *Cross-Encoder* rulat intern (ex: `ms-marco-MiniLM-L-6-v2`) pentru a evita limitările de rețea.
2. **Memory Bottleneck:** Scanarea unui DataFrame Pandas în memorie RAM cedează la milioane de înregistrări. **Soluția:** Migrarea către o bază de date vectorială dedicată (ex: Pinecone), care permite *Hybrid Search nativ* în milisecunde.

**When might the system produce "confident but incorrect" results?**
Sistemul devine periculos de încrezător și greșit atunci când datele tabulare intră în contradicție cu textul descriptiv, exploatând logica mea de *Soft-Fail*. 
*Exemplu teoretic:* O corporație gigant (peste 10.000 de angajați) are accidental valoarea `NaN` la coloana `employee_count`. Modulul 2 o lasă să treacă din cauza Soft-Fail-ului. Dacă în descrierea sa (textul liber) compania folosește cuvinte precum "agile, startup-like culture, innovation", LLM-ul din Modulul 3, neavând cifra reală de angajați, îi va acorda 95/100 cu mare încredere pentru o căutare de "startups". LLM-ul este încrezător în evaluarea textului, dar textul în sine a indus în eroare sistemul.

**What would I monitor in production to detect these failures?**
Pentru a detecta acest *Data Drift* și halucinațiile modelului în producție, aș implementa trei piloni de monitorizare:
1. **Implicit User Feedback (Click-Through Rate):** Aș monitoriza telemetria utilizatorilor. Dacă sistemul returnează un candidat cu scor de 98/100, dar utilizatorii dau *scroll* peste el în 90% din cazuri, este un *False Positive* asigurat, declanșând o revizuire a promptului din LLM.
2. **Embeddings-to-Score Divergence:** Aș seta alerte tehnice pentru anomaliile matematice. Dacă distanța (Cosine Similarity) din Modulul 2 este slabă (ex: sub 0.3), dar LLM-ul dă un scor de peste 90, este un indicator clasic de halucinație a LLM-ului care ignoră faptele matematice de bază.
3. **LLM Confidence Variance Testing:** Aș rula zilnic un set automat de interogări pe un "Golden Dataset" (un set mic de date cu rezultate cunoscute și validate uman). Dacă scorurile pentru aceleași companii fluctuează brusc de la o zi la alta, aș ști că s-au făcut modificări "sub capotă" în parametrii modelului provider-ului.