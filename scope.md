# EE Engineering Copilot: Scope

## Purpose
Answer Year III, Part I EE study questions from approved class material only.
Cite file and page for every answer. Refuse when the material has no evidence.

## Users
About 50 students of one class. One maintainer.

## In scope
Subjects: dsp, dcs, swp, emd, tht, pe (see data/scope.json).
Question types: definitions, derivations, numerical problems (setup only in
Part I), lab and viva questions.
A question is in scope when its answer can be found on a page of a
registered, shareable document in the corpus.

## Out of scope (refuse)
- Course ENEE 301 (Project Engineering and Management).
- Other courses, general knowledge, news, personal or medical advice.
- Real design or protection settings for real equipment.
- Writing assignments or code unrelated to the six subjects.
- Questions about the exam paper or predicted marks.
- Content that needs images, handwriting or scanned pages (Part I).

## Boundary rule
The topic can be in scope while the answer is missing from the corpus.
In that case the copilot refuses with the standard sentence. It never
answers from model memory.

## Answer contract
1. Use only the retrieved context.
2. Cite every claim as [source, page].
3. Give units for every number. Show formula, values and result.
4. Refusal sentence (exact): "I could not find this in the documents."
5. Never present output as a design authority.

## Data rules
- Index only files with shareable=yes in the class index.
- Textbooks are shareable=no.
- Eval questions and past papers stay in eval/source/, not in the corpus.
- Send only shareable text to the API provider.

## Targets (Week 1)
R1 citations 100% | R2 refusal >= 90% | R3 P@5 >= 0.70 | R4 faithfulness >= 4.0
R6 >= 5 documents per subject | R8 >= 8 eval questions per subject

## Change control
Scope changes need a version bump and a re-run of the eval. The eval set
is frozen after Day 6.