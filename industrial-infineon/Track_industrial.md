# Track: Industrial AI
## Use Case Title: Prozesslogik lernen und benchmarken

**Challenge Owner:** TBD
**Mentor(en):** TBD
**Schwierigkeitsgrad:** Fortgeschritten bis Expert
**Geschätzter Scope:** Ja, der Case ist in 36h realistisch umsetzbar, weil mit synthetischen Daten gearbeitet wird, die Modellwahl offen ist und bereits ein reduzierter Umfang mit Datengenerierung, einem Trainingslauf und einer belastbaren Evaluation einen sinnvollen Abschluss bildet.

---

### 1. Problem Statement (3–5 Sätze)
In vielen industriellen Prozessen lassen sich Abläufe als lange Sequenzen von Schritten beschreiben, deren Bedeutung stark von Reihenfolge, Zwischenschritten und Prozesslogik abhängt. Auch in der Halbleiterfertigung entstehen Produkte über komplexe Prozessrouten, in denen Materialien aufgetragen, strukturiert, verändert und wieder entfernt werden. Der Hackathon abstrahiert dieses Problem und untersucht, wie gut Modelle solche Prozesssequenzen lernen, nächste Schritte vorhersagen und auf veränderte oder ungesehene Abläufe generalisieren können. Entscheidend ist dabei die Frage, ob ein Modell nur bekannte Muster reproduziert oder ein übertragbares Verständnis der zugrunde liegenden Prozesslogik aufbaut.

### 2. Warum das wichtig ist (Business Context)
- Die Herausforderung besteht darin, dass industrielle Prozessabläufe zwar stark strukturiert sind, diese Struktur aber ohne sauberes Training und Benchmarking nur schwer robust modellierbar und vergleichbar ist.
- Davon profitieren vor allem Teams aus Prozessentwicklung, ML Engineering und Forschung, die systematisch untersuchen wollen, welche Modell- und Datenstrategien für prozessartige Sequenzen funktionieren.
- Der Bezug zu **European AI Sovereignty** liegt im Fokus auf eigener Datengenerierung, Training auf dem Cluster, reproduzierbaren Evaluationspipelines und dem Vergleich unterschiedlicher Modellgrößen. Im Vordergrund steht also belastbare Infrastruktur- und Modellkompetenz statt eines reinen API- oder Wrapper-Ansatzes.

### 3. Erwartetes Outcome / Definition of Done
- **Minimum Viable Result:** Ein nachvollziehbarer End-to-End-Workflow mit synthetischer Datengenerierung, mindestens einem trainierten Modell, einem Baseline-vs.-Post-Training-Vergleich und einem klar dokumentierten Benchmark für Prozesssequenzen.
- **Stretch Goals:** Vergleich mehrerer Modellgrößen oder Architekturen, Analyse von Skalierungseffekten, Generalisierung auf ungesehene oder abgewandelte Sequenzen, Einbezug optionaler Prozessparameter und ein kleiner Demonstrator.
- **Lernziele für Teilnehmer:innen:** Aufsetzen und Trainieren von Sequenzmodellen auf Cluster-Infrastruktur, Generierung synthetischer Trainingsdaten, systematisches Benchmarking, Analyse von Skalierungseffekten sowie sauberes Evaluieren von Vorher-/Nachher-Verhalten.
- **Format:** Trainings-Run mit Eval-Report; optional ergänzt durch Demo, Prototyp oder Demonstrator.
- **Demonstrator Stretch Goal:** Eine kleine Vorher-/Nachher-Demo, die Beispielausgaben eines Baseline-Modells und eines trainierten Modells nebeneinander zeigt, z.B. bei Next-Step-Prediction, Sequenzvervollständigung oder der Erkennung untypischer Prozessschritte.

### 4. Modell-Spezifikation
- **Typ:** Frei wählbar. Geeignet sind insbesondere Transformer-basierte Sequenzmodelle, LLMs, andere Sequenzmodelle oder hybride Ansätze.
- **Basismodell:** Frei wählbar, z.B. From-Scratch auf synthetischen Daten oder ein offenes Basismodell als Ausgangspunkt.
- **Modellgrößen:** Eine oder mehrere Größen sind ausdrücklich erwünscht, um Small-vs.-Large-Setups und Skalierungseffekte vergleichen zu können.
- **Trainingsmethoden:** Frei wählbar. Naheliegend sind Next-Step-Prediction, Sequenzvervollständigung, SFT sowie RL-basierte Ansätze wie GRPO oder andere zum Modell passende Trainingsziele.
- **Constraints:** Fokus auf reproduzierbare Trainings- und Evaluations-Setups auf Cluster-Infrastruktur sowie auf einem offenen, nachvollziehbaren Stack statt Black-Box-API-Lösungen.

### 5. Aufgabenstruktur (Levels)
- **Level 1:** Ausgangsdaten verstehen, zusätzliche synthetische Daten erzeugen und eine belastbare Baseline für Next-Step-Prediction oder Sequenzvervollständigung aufsetzen.
- **Level 2:** Ein Modell trainieren, anschließend gezielt weiter tunen oder verbessern und den Unterschied zwischen Baseline, trainiertem Modell und nachträglich optimiertem Modell mit einem eigenen Benchmark für Prozessverständnis sichtbar machen.
- **Level 3 / Stretch:** Skalierungseffekte systematisch analysieren, z.B. durch den Vergleich von Modellgrößen, Compute Time und Datenmenge, sowie deren Einfluss auf Leistung und Generalisierung. Optional können zusätzlich verschiedene Architekturen oder Prozessparameter einbezogen werden.

### 6. Daten & Ressourcen
- **Datensätze:** Es liegen Daten für drei Produktfamilien vor: IC, IGBT und MOSFET. Pro Familie gibt es eine Long-Description-Variante mit `STEP` und `DESCRIPTION`, eine Variante mit zusätzlichen `REALISTIC FAB-LEVEL PARAMETERS` sowie eine synthetische Sequenzdatei.
- **Datenformat:** CSV-Dateien mit prozessbezogenen Schrittfolgen. Je nach Datei enthalten die Daten nur `STEP` oder zusätzlich textuelle Beschreibungen und realistische fab-nahe Parameter pro Prozessschritt. Die vorgenerierten Trainingssequenzen liegen im **Langformat** vor (`SEQUENCE_ID, STEP`; eine Zeile pro Schritt).
- **Datenmenge:** Im Ordner `training_data/` stehen je **1.000 vorab generierte, validierte Sequenzen** pro Produktfamilie bereit (3.000 Sequenzen gesamt, je ca. 115–150 Schritte). Zusätzlich liegen die originalen neun Referenz-CSV-Dateien mit rund 1.100 Datenzeilen im Hauptverzeichnis vor. Der kombinatorische Raum gültiger Sequenzen ist sehr groß (MOSFET ~51 Mrd., IGBT ~13 Bio., IC ~6 Mrd. unterschiedliche Sequenzen), sodass Teams mit dem beigelegten Skript `training_data/generate_sequences.py` beliebig viele weitere Trainingssequenzen erzeugen können. Die zugrundeliegende Prozessgrammatik, alle Validierungsregeln und die Eval-Protokoll-Spezifikation sind in `training_data/generation_rules.md` dokumentiert.
- **APIs/Systeme:** Keine zwingenden APIs vorgegeben; relevant sind vor allem die bereitgestellten CSV-Beispieldaten und die Cluster-Umgebung.
- **Compute:** Training auf dem Leonardo Cluster ist explizit vorgesehen. GPU-Quota pro Team: TBD.
- **NDAs / Datenschutz:** Datenschutz und Freigabe der bereitgestellten Daten sind geklärt; die Verwendung im Rahmen des Hackathons ist in Ordnung.

### 7. Evaluation & Benchmarking
- **Eval-Setup:** Es steht ein gemeinsames, festes Eval-Set bereit, das von den Veranstaltern verteilt wird. Es umfasst zwei Teilmengen:
  - **Next-Step- und Completion-Aufgaben** (`eval_input_valid.csv`): 600 Einträge – je 100 zurückgehaltene Sequenzen pro Familie, jeweils bei 60 % und 80 % Vervollständigung abgeschnitten.
  - **Anomalieerkennung** (`eval_input_anomaly.csv`): 987 gemischte Sequenzen – 387 mit gezielt injizierten Prozessregelverstößen (nach 10 Regeltypen beschriftet) und 600 gültige Sequenzen, ungeordnet und ohne Labels.
- **Drei Submission-Aufgaben (Teams reichen Ergebnisse ein):**

  | #   | Aufgabe                      | Eingabe                                      | Metrik(en)                                                                                   |
  | --- | ---------------------------- | -------------------------------------------- | -------------------------------------------------------------------------------------------- |
  | 1   | **Next-Step-Prediction**     | Partielle Sequenz                            | Top-1-Accuracy, Top-3-Accuracy, Top-5-Accuracy, MRR                                          |
  | 2   | **Sequenzvervollständigung** | Partielle Sequenz (60 % oder 80 %)           | Exact Match Rate, Normalized Edit Distance, Token Accuracy, Block-level Accuracy             |
  | 3   | **Anomalieerkennung**        | Vollständige Sequenz (mit/ohne Regelverstoß) | Binary Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC, Rule Attribution Accuracy |

- **Zusätzliches Generalisierungs-Reporting (nur durch Veranstalter, nach Abgabe):**

  | #   | Aufgabe                 | Eingabe                                          | Metrik(en)                                                |
  | --- | ----------------------- | ------------------------------------------------ | --------------------------------------------------------- |
  | 4   | **Generalisierung OOD** | ID- vs. OOD-Split auf unbekannter Produktfamilie | Performance-Drop ID → OOD (pro Hauptmetrik aus Tasks 1–3) |

  Teams reichen für Task 4 keine separaten Ergebnisse ein. Die Veranstalter wenden die eingereichten Modelle nach der Abgabe auf den OOD-Datensatz an und berechnen den Performance-Drop.

- **Scoring:** Das Bewertungsskript `eval_metrics.py` ist für alle drei Aufgaben einsatzbereit und benötigt keine externen Abhängigkeiten. Es gibt pro Aufgabe einen detaillierten Report mit Aufschlüsselung nach Familie und Schnittpunkt.
- **Generalisierung (Task 4):** Die Generalisierungsfähigkeit wird zusätzlich an einer vierten, im Training nicht enthaltenen und nicht kommunizierten Produktfamilie bewertet. Diese Familie ist den Teilnehmenden nicht bekannt und wird ausschließlich nach Abgabe durch die Veranstalter zur Auswertung genutzt. Bewertet wird der Performance-Drop (ID → OOD) über alle Hauptmetriken.
- **Test-Frequenz:** Sinnvoll sind automatisierte Evaluationsintervalle während des Trainings, um Lernverlauf, Overfitting und Skalierungseffekte sichtbar zu machen. Die konkrete Frequenz definieren die Teams selbst.
- **Inference-Stack:** Eine UI ist optional. Für Live-Tests reicht bereits ein einfacher Inference-Workflow oder Notebook-basierter Demonstrator. Besonders interessant ist eine direkte Gegenüberstellung von Baseline-Output und trainiertem Modell-Output auf identischen Eingaben.
- **Visualisierung:** Erwartet werden mindestens Loss-Kurven, Leistungsmetriken über die Zeit und vergleichende Darstellungen zwischen Baseline und trainiertem Modell; weitere Visualisierungen sind willkommen.
- **Vergleichbarkeit:** Alle Teams arbeiten mit demselben Eval-Set und denselben Metriken; die Ergebnisse sind damit direkt vergleichbar.

**Beispiel für Demonstrator-Outputs:**

- **Baseline-Modell:** Eingabe: `RECEIVE WAFER LOT -> LOT IDENTIFICATION -> INITIAL WAFER INSPECTION -> ?` Ausgabe: `ETCH` oder ein generischer, fachlich unplausibler nächster Schritt.
- **Trainiertes Modell:** Eingabe: `RECEIVE WAFER LOT -> LOT IDENTIFICATION -> INITIAL WAFER INSPECTION -> ?` Ausgabe: `MEASURE THICKNESS` oder `MEASURE INITIAL THICKNESS` mit höherer Plausibilität im Kontext der Prozessfolge.
- **Baseline-Modell:** Unvollständige Sequenz mit fehlendem Reinigungsschritt wird ohne erkennbare Prozesslogik ergänzt.
- **Trainiertes Modell:** Erkennt, dass vor einem Strukturierungs- oder Depositionsschritt ein plausibler Vorbereitungs- oder Reinigungsschritt fehlt und ergänzt die Sequenz entsprechend.

### 8. Technische Hinweise
- Vorgeschlagener Tech Stack: Python, PyTorch oder vergleichbare Frameworks für Sequenzmodelle, ergänzt um sauberes Experiment-Tracking und Evaluationsskripte.
- Bekannte Stolpersteine: Qualität und Verteilung synthetischer Daten, fairer Vergleich zwischen Modellen, sinnvolle Generalisierungstests, sauberes Checkpointing auf Cluster-Infrastruktur und ein Benchmark, der mehr als reines Auswendiglernen misst.
- Bekannte Baseline: Die bereitgestellten Beispieldaten dienen als Ausgangsbasis; eine konkrete interne Modell-Baseline ist aus dem vorliegenden Briefing nicht ersichtlich und sollte bei Bedarf ergänzt werden.

### 9. Bewertungskriterien (track-spezifisch)
- Technische Tiefe und nachvollziehbare Modell- und Datenentscheidungen
- Qualität des Trainings- und Benchmark-Setups auf echter Infrastruktur
- Reproduzierbarkeit und Klarheit der Evaluation
- Aussagekraft des Vergleichs zwischen Baseline, trainiertem Modell und optionalen Skalierungsvarianten
- Qualität von Demo, Visualisierung und Ergebnisaufbereitung

### 10. Kontakt & Support während des Events
- Mentor vor Ort: Simeon