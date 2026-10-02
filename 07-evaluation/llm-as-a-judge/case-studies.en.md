# Applying judges to RAG, agents, and memory

[中文](case-studies.md) · **English**

“The response looks good” misses different problems in different systems. All examples here are fictional. The questions are: **what evidence is needed, who can check it, and what conclusion is justified?**

## RAG: separate correctness, relevance, and grounding

The supplied policy allows returns only for unopened products within 7 days.

| Variant | Treatment |
| --- | --- |
| Friendly response promises 30 days | Grounding fails; record tone separately |
| Accurate policy quotation ignores the user's opened product | Grounding may pass; task completeness may fail |
| Retrieval returns no relevant policy | Record retrieval failure; policy-dependent judgment is unknown |
| Reference says 30 days, evidence says 7 | Flag reference conflict and review the dataset |
| Response cites nonexistent policy-9 | Check ID existence programmatically; still verify semantic support |

Missing retrieval and incorrect generation despite available evidence are different component failures. Combining them into one low score obscures where to intervene.

A citation's existence doesn't establish support. Check whether the cited passage entails the associated claim.

## Agents: saying “Done” doesn't establish completion

The user asks: “Find two flights, but don't book.” The agent presents results and says it's done, yet the trace contains a create_booking call.

This is a permission violation, not a low helpfulness score that can be averaged away.

Another version avoids booking but claims to have saved a draft when no draft exists in the final state. Inspect the state:

~~~python
checks = {
    "no_booking": not any(event["type"] == "create_booking" for event in events),
    "draft_saved": final_state.get("draft_id") is not None,
}
~~~

These illustrate specific rules; a real system needs trustworthy event records and state reads. An accepted tool request may not mean its side effect has completed. Include delays, retries, and idempotency in scenarios.

A judge can assess whether the failure explanation is honest, the proposed next step is usable, or a user constraint was overlooked, citing relevant evidence. Reset environments and record tool failures and final states so environment problems aren't mistaken for model regression. See [Anthropic's agent-evaluation guidance](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents).

## Memory: remembering isn't always appropriate

Old memory: “I prefer short answers.” Current instruction: “Explain the derivation in detail this time.”

A one-sentence response shouldn't earn credit just for using memory. Check:

- **Current intent:** does the explicit current request take priority?
- **Source and time:** did the user state the preference, or did the model infer it? Has it changed?
- **Correction persistence:** does an update affect later turns and sessions?
- **Privacy and boundaries:** is unnecessary or inappropriate memory exposed?

Longitudinal evaluation needs time-ordered state updates. Resetting memory every turn doesn't test lasting corrections. User simulators can create stress scenarios, but simulated satisfaction isn't evidence of real user satisfaction. Sharing an unvalidated preference model between scenario generation and grading isn't independent confirmation.

## Code: execute what can be executed

Compilation, unit tests, and static checks provide a useful first layer. A judge can inspect readability, requirement coverage, and potential edge cases; saying “correct” doesn't replace execution.

Tests may miss requirements too. An implementation that crashes on empty input can pass a normal-input-only suite. Add boundary and adversarial inputs, then inspect whether the tests themselves encode the relevant constraints. Don't rely on one model's approval for security-critical behavior.

## Multimodal tasks: did the judge receive the necessary information?

A screenshot contains a price table and the answer picks the wrong column. If the judge receives OCR that lost the row-column relationship, a stronger model cannot recover evidence that was omitted.

Record resolution, cropping, OCR, and image-text ordering. Compare text-only, image-only, combined, and conflicting inputs. Decide which questions should become harder when the image is removed, and which should be unaffected.

This tests use of visual evidence, not the presence of a multimodal label. Allow unknown when crucial text is unreadable instead of forcing a precise factual verdict.

## A result should point back to a component

~~~text
retrieval: necessary material was not found
generator: evidence was present, but the deadline was wrong
agent: permission violation or incomplete final state
memory: stale preference overrode the current request
judge: missed failure, bad citation, invalid output, or uncertainty
~~~

All can lower an aggregate score, but the fixes differ. Separate records tell us where to work next.

Next: [Run a minimal evaluation workflow](implementation.en.md)
