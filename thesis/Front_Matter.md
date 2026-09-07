# FRONT MATTER (εμπροσθόφυλλα)

## Προτεινόμενος τίτλος

**ΑΝΑΠΤΥΞΗ ΕΥΦΥΟΥΣ EVENT-DRIVEN ΣΥΣΤΗΜΑΤΟΣ ΠΡΟΒΛΕΨΗΣ ΧΡΗΜΑΤΙΣΤΗΡΙΑΚΩΝ ΚΙΝΗΣΕΩΝ ΜΕ ΧΡΗΣΗ ΤΕΧΝΗΤΗΣ ΝΟΗΜΟΣΥΝΗΣ ΚΑΙ ΑΝΑΛΥΣΗΣ ΣΥΝΑΙΣΘΗΜΑΤΟΣ**

*(Αγγλικά: Development of an Intelligent Event-Driven Stock Market Prediction System Using Artificial Intelligence and Sentiment Analysis)*

---

## Δήλωση πνευματικών δικαιωμάτων και χρήσης AI

Το σύνολο της εργασίας αποτελεί πρωτότυπο έργο και δεν παραβιάζει δικαιώματα τρίτων. Στο πλαίσιο της διαφάνειας, δηλώνεται ρητώς η χρήση εργαλείων τεχνητής νοημοσύνης κατά την εκπόνηση. Συγκεκριμένα, αξιοποιήθηκε το εργαλείο βοηθού κώδικα **Claude Code (Anthropic)** στις εξής εργασίες: (α) υποστήριξη στην ανάπτυξη και αποσφαλμάτωση (debugging) τμημάτων του κώδικα, (β) γλωσσική επιμέλεια και βελτίωση της δομής του κειμένου, και (γ) βοήθεια στη μορφοποίηση του τελικού εγγράφου. Δεν χρησιμοποιήθηκε για την παραγωγή ερευνητικών αποτελεσμάτων, αριθμητικών δεδομένων ή συμπερασμάτων· όλα τα ποσοτικά αποτελέσματα προέκυψαν από την εκτέλεση του πραγματικού κώδικα και επαληθεύτηκαν από τον συγγραφέα. Κάθε παραγόμενο απόσπασμα κώδικα ή κειμένου ελέγχθηκε, δοκιμάστηκε και τελεί υπό την πλήρη ευθύνη του συγγραφέα.

---

## Περίληψη

Η παρούσα διπλωματική εργασία αφορά την ανάπτυξη ενός έξυπνου, event-driven συστήματος πρόβλεψης χρηματιστηριακών κινήσεων, το οποίο συνδυάζει αυτοματοποιημένη συλλογή δεδομένων, ανάλυση συναισθήματος και μηχανική μάθηση σε ένα ενιαίο περιβάλλον. Το σύστημα αντλεί δεδομένα από επτά ετερογενείς πηγές — χρηματιστηριακές πλατφόρμες, ειδησεογραφικές ιστοσελίδες και κοινωνικά δίκτυα — τα εμπλουτίζει με μεταδεδομένα αξιοπιστίας και αναλύει το συναίσθημα του ειδησεογραφικού περιεχομένου με το μοντέλο FinBERT. Πέντε μοντέλα μηχανικής μάθησης (Random Forest, XGBoost, LightGBM, Logistic Regression και νευρωνικό δίκτυο LSTM) συνδυάζονται μέσω συνδυαστικής μάθησης (ensemble learning) για την παραγωγή προτάσεων «Αγορά», «Πώληση» ή «Κράτηση», ενώ τα αποτελέσματα παρουσιάζονται μέσω διαδραστικής διεπαφής που περιλαμβάνει προβολή τιμής και ανάλυση κινδύνου. Η αξιολόγηση πραγματοποιήθηκε με αυστηρό χρονικό διαχωρισμό των δεδομένων ανά δείκτη, ώστε να αποφευχθεί η διαρροή πληροφορίας. Σε σύνολο ελέγχου 4.452 παρατηρήσεων, τα ισχυρότερα μοντέλα πέτυχαν ακρίβεια έως **41.7%** (LSTM)· για το ισχυρότερο μοντέλο με πλήρη στατιστική ανάλυση (Logistic Regression, **41.6%**) το 95% διάστημα εμπιστοσύνης είναι 40.1–43.0%. Η ακρίβεια είναι στατιστικά σημαντικά υψηλότερη τόσο του τυχαίου ορίου (33.3%· διωνυμικός έλεγχος) όσο και του ταξινομητή πλειοψηφίας (37.9%· συζευγμένος έλεγχος McNemar, p < 0.001). Ειδική μελέτη απομόνωσης (ablation) τεκμηρίωσε τη θετική συνεισφορά της ανάλυσης συναισθήματος (+2.0 ποσοστιαίες μονάδες ακρίβειας). Το σύνολο του συστήματος υλοποιήθηκε αποκλειστικά με τεχνολογίες ανοιχτού κώδικα, αποδεικνύοντας τη δυνατότητα αξιοποίησης της τεχνητής νοημοσύνης σε ρεαλιστικές χρηματοοικονομικές εφαρμογές χωρίς εμπορικά εργαλεία.

**Λέξεις κλειδιά:** πρόβλεψη χρηματιστηρίου, μηχανική μάθηση, ανάλυση συναισθήματος, FinBERT, συνδυαστική μάθηση, τεχνητή νοημοσύνη

---

## Abstract

This thesis concerns the development of an intelligent, event-driven stock market prediction system that combines automated data collection, sentiment analysis, and machine learning in a single environment. The system gathers data from seven heterogeneous sources — financial platforms, news websites, and social networks — enriches it with reliability metadata, and analyzes the sentiment of news content using the FinBERT model. Five machine learning models (Random Forest, XGBoost, LightGBM, Logistic Regression, and an LSTM neural network) are combined through ensemble learning to produce "Buy", "Sell", or "Hold" recommendations, while the results are presented through an interactive interface that includes price forecasting and risk analysis. The evaluation was carried out using a strict per-ticker chronological train-test split to prevent data leakage. On a test set of 4,452 observations, the strongest models reached up to **41.7%** accuracy (LSTM); for the strongest model with full statistical analysis (Logistic Regression, **41.6%**), the 95% confidence interval is 40.1–43.0%. This accuracy is statistically significantly higher than both the random baseline (33.3%; binomial test) and the majority-class classifier (37.9%; paired McNemar test, p < 0.001). A dedicated ablation study documented the positive contribution of sentiment analysis (+2.0 percentage points of accuracy). The entire system was implemented exclusively with open-source technologies, demonstrating the potential of artificial intelligence in realistic financial applications without commercial tools.

**Keywords:** stock market prediction, machine learning, sentiment analysis, FinBERT, ensemble learning, artificial intelligence

---

## Ευχαριστίες

Η παρούσα διπλωματική εργασία εκπονήθηκε στο πλαίσιο ολοκλήρωσης των προπτυχιακών μου σπουδών στο Τμήμα Ηλεκτρολόγων Μηχανικών και Μηχανικών Υπολογιστών του Πανεπιστημίου Πελοποννήσου.

Θα ήθελα να ευχαριστήσω θερμά τον επιβλέποντα καθηγητή μου, **Αναπληρωτή Καθηγητή Σωτήρη Χριστοδούλου**, για την πολύτιμη καθοδήγηση και τις συμβουλές του καθ' όλη τη διάρκεια της εκπόνησης της εργασίας. Επίσης, ευχαριστώ όλους τους καθηγητές της σχολής για τις γνώσεις που μου προσέφεραν κατά τη διάρκεια των σπουδών μου. Τέλος, ευχαριστώ την οικογένειά μου και τους φίλους μου για τη διαρκή στήριξή τους.

---

## Κατάλογος Συντομογραφιών

| Συντομογραφία | Επεξήγηση |
|---|---|
| AI | Τεχνητή Νοημοσύνη (Artificial Intelligence) |
| API | Διεπαφή Προγραμματισμού Εφαρμογών (Application Programming Interface) |
| BERT | Bidirectional Encoder Representations from Transformers |
| CSV | Comma-Separated Values |
| DL | Βαθιά Μάθηση (Deep Learning) |
| EMH | Υπόθεση Αποτελεσματικής Αγοράς (Efficient Market Hypothesis) |
| ES | Expected Shortfall |
| ETF | Διαπραγματεύσιμο Αμοιβαίο Κεφάλαιο (Exchange-Traded Fund) |
| FinBERT | Financial BERT |
| GARCH | Generalized Autoregressive Conditional Heteroskedasticity |
| HTML | HyperText Markup Language |
| JSON | JavaScript Object Notation |
| LLM | Μεγάλο Γλωσσικό Μοντέλο (Large Language Model) |
| LSTM | Long Short-Term Memory |
| MACD | Moving Average Convergence Divergence |
| ML | Μηχανική Μάθηση (Machine Learning) |
| NLP | Επεξεργασία Φυσικής Γλώσσας (Natural Language Processing) |
| REST | Representational State Transfer |
| RSI | Relative Strength Index |
| RSS | Really Simple Syndication |
| SHAP | SHapley Additive exPlanations |
| UI | Διεπαφή Χρήστη (User Interface) |
| VaR | Value at Risk |
| XML | eXtensible Markup Language |
