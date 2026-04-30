# Synthetic Data Generation Notes

This dataset is artificial and was created only for the Assignment 2 proof of concept.
It is designed to look like Sistrade-style ERP support tickets, but it must not be
interpreted as real Sistrade operational data.

Generation choices:
- fixed random seed for reproducibility;
- fake client names and fake support resolvers;
- Portuguese and English mixed text;
- ERP vocabulary for invoicing, stock, production, accounting, CRM, reporting,
  permissions, integrations, and general system issues;
- intentionally noisy descriptions such as tired-user comments, incomplete context,
  and ambiguous notes like "not sure if it is a bug or configuration";
- previous classifications are sometimes wrong to represent realistic manual triage.

The labels are deterministic because they are assigned by the generator rules. This
makes the data useful for demonstrating an ML workflow, but it does not prove real
performance on real customer tickets.
