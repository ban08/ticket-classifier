# Presentation Outline

## Slide 1 - Title

Sistrade Ticket Classification ML POC.

Say: This project is a university proof of concept showing how classic ML can assist support ticket triage.

## Slide 2 - Assignment Goal

Show the official goal: identify a realistic problem, build a model with artificial data, build a web app, and deliver code/documentation.

Say: We focused on a small realistic problem rather than a full enterprise solution.

## Slide 3 - Company Context

Explain Sistrade support tickets and ERP modules such as invoicing, stock, production, CRM, accounting, reporting, permissions, and integrations.

Say: Support tickets need classification, priority, and routing before they can be handled.

## Slide 4 - Business Problem

Show examples of wrong classification.

Say: Wrong routing wastes time, delays urgent issues, and interrupts the wrong teams.

## Slide 5 - Why Not A Chatbot

List hallucination risk, company-specific knowledge, evaluation difficulty, business risk, and paid API concerns.

Say: For this assignment, classification is safer, measurable, and more aligned with a POC.

## Slide 6 - ML Task

Show predicted labels: ticket type, priority, target team, module, and component.

Say: The model suggests structured triage fields for human review.

## Slide 7 - Synthetic Dataset

Describe 1,400 generated tickets with fake clients, noisy text, Portuguese/English phrases, urgency clues, and wrong previous classifications.

Say: The data is artificial, so it demonstrates the workflow but does not prove real performance.

## Slide 8 - Model Approach

Show TF-IDF text features and Logistic Regression classifiers.

Say: We used deterministic local ML with scikit-learn, no LLMs and no paid APIs.

## Slide 9 - Evaluation

Show the metric table.

Say: Type and priority perform very well on synthetic data; team and component are harder and reveal the limits of the POC.

## Slide 10 - Web App Demo

Open the Streamlit app and enter an urgent invoicing bug.

Say: The app shows predicted fields, confidence scores, and deterministic keyword evidence.

## Slide 11 - Limitations

Mention synthetic data, generator bias, no real integration, no automatic answering, and no production guarantees.

Say: This is a demonstration of feasibility and workflow, not a deployable system.

## Slide 12 - Future Work

Suggest anonymized historical data, feedback loops, thresholding, model tuning, and integration after validation.

Say: The next step would be validating with real tickets and support staff.
