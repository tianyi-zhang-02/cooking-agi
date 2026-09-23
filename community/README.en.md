# Building these notes together

[中文](README.md) · **English**

These started as one person's study notes. We're opening them up for more people to write and maintain together. Fix a sentence or look after a whole topic: clear work matters more than elaborate process.

**The short version: find a page → ask a question or open a PR → have another person review it → merge after checks pass.**

## Where to start

| What you want to do | First step |
| --- | --- |
| Fix a typo, reference, or translation | Open a small PR; no permission needed first |
| Question an explanation or add an example | Use “Discuss & contribute” below the article; you don't need to know the answer yet |
| Write a new note or interactive diagram | Open a [proposal](https://github.com/tianyi-zhang-02/cooking-agi/issues/new?template=proposal.yml) with the problem, location, and approach, then a draft PR |
| Help maintain a topic | Propose a scope and rough availability; daily attendance is not expected |
| Add a credit or an optional map location | Use the [contribution record](https://github.com/tianyi-zhang-02/cooking-agi/issues/new?template=add-me-to-the-crew.yml) and link an issue or PR |

Discussions live in GitHub Issues; reviews of specific edits live in PRs. Chinese and English versions share a discussion entry. Search first and continue an existing thread when possible. We won't maintain a separate comments system.

## Who looks after each area

<div data-collaboration-areas></div>

**An open area still has a contact**: repository maintainers handle reviews until someone volunteers and confirms. Only consenting reviewers are listed. We don't assign work to friends in advance or award access based on commit counts.

Authors own their content; area reviewers check accuracy, explanations, and validation; maintainers handle cross-area consistency, releases, and unresolved decisions. These are responsibilities, not ranks. Anyone can discuss or review, but a formal merge needs an authorized reviewer.

## How a change gets reviewed

1. **Keep it focused.** Aim for one problem per PR. List affected areas and involve their reviewers when a change crosses boundaries.
2. **Show the basis.** Cite public sources, give reproduction conditions, and separate observations, hypotheses, and validated claims. Passing checks does not establish correctness.
3. **Get another human's review.** Routine changes need at least one relevant CODEOWNER approval from someone other than the author. Address or explicitly resolve review comments before merging; new commits invalidate earlier approvals.
4. **Pass automated checks.** Build, privacy scan, collaboration checks, and tests must pass. Bilingual structure checks are currently advisory, not proof of translation quality. A one-language change needs a follow-up plan in its PR.
5. **Merge, then publish.** Preserve authors and co-authors. Only main deploys; a draft PR is not a publication.

There is currently only one confirmed maintainer. Administrators temporarily retain an emergency bypass for small fixes without another available reviewer or release failures. Explain its use in the PR; self-checking is not independent review. Revisit this exception once a second reviewer confirms. Written rules and GitHub's active protections are separate; maintainers should verify repository settings.

## When we disagree

Typos don't need a vote. For disputed methods or claims, compare sources, assumptions, and experiments rather than seniority. If the evidence is inconclusive, explain both interpretations and the uncertainty.

**Reserve a recorded decision for structural changes, public interfaces, review rules, or permissions:**

- Open a proposal describing the problem, at least one alternative, impact, and rollback. Normally allow **72 hours** for input. Silence is not consent.
- If multiple maintainers are available, involve another relevant maintainer. With one maintainer, record it as a sole-maintainer decision rather than claiming consensus.
- If discussion remains unresolved, a repository maintainer without a conflict makes the decision. For personal conflicts, ask a neutral participant to help. Record the decision, reasoning, dissent, and conditions for revisiting it in the original issue.
- For a privacy exposure or site failure, hide, revert, or contain the problem first and document afterward. Don't post sensitive details in a public issue or repeat them to explain the incident.

No committee or weekly meetings required. Ordinary content uses the normal PR process.

## Contributions beyond commits

<div data-community-credits></div>

[Behind the notes](../contributors.en.md) keeps the astronauts, rolling credits, and commit history. Commit counts describe only part of the activity; they are neither a quality ranking nor voting rights. Preserve co-author credit for shared writing. Helpful reviews and corrections in issues can be acknowledged above with the person's consent. Map locations are optional and can be removed on request.

## A few boundaries

- Discuss the work, not the person. Different views are welcome; harassment, personal attacks, ads, and off-topic fights are not.
- Use public sources or publicly reproducible examples. Don't submit internal company data, recruiting details, credentials, or other people's private information. If unsure, don't post it. Automated scanning is only a backstop.
- AI can help with writing, translation, or code, but the author must verify it. Describe material AI assistance in the PR; don't fabricate citations, experiments, or independent reviews.
- Credit sources and confirm permission to reuse figures and code. Explain new external scripts, dependencies, or third-party services before adding them.
- Usually explain the problem and give a warning first; restrict participation for repeated or serious abuse. Safety issues may require immediate action. Don't erase valid past contributions over disagreements; handle removal requests for privacy or documented abuse.

For writing conventions and local preview, see the [contributing guide](../CONTRIBUTING.md#english).
