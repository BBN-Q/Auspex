# Engineering principles: convivial software

## Purpose

Inspired by Ivan Illich's *Tools for Conviviality*, software should extend users' capabilities, understanding, and agency. It must not create unnecessary dependence on opaque systems, institutions, developers, or LLMs. Automation can be valuable; loss of human judgment and control is not an inevitable cost of automation. 

We aim for software as a crafted instrument: fitted to its purpose, honest about its limits, understandable in its construction, and capable of being maintained and adapted by others. Automation may assist its making, but does not remove responsibility for its quality. We accept the costs of careful engineering: time spent understanding, discussing, verifying, documenting, and sometimes choosing not to build. 

## Agency and intelligibility

- Give users meaningful opportunities to inspect, direct, reproduce, and where appropriate override consequential behavior.
- Expose assumptions, operating limits, intermediate results, uncertainty, and failure modes at an appropriate level. Observability is for understanding, not for log volume.
- Prefer useful abstractions that clarify mechanisms. Avoid abstractions that hide consequential assumptions or make substitution impossible.
- Prefer standard, inspectable data representations and composable interfaces when they address real needs.

## Human-readable implementation

- A competent engineer should be able to understand, debug, modify, and verify the software with ordinary tools and without an LLM or the original conversation.
- Prefer explicit data flow, familiar idioms, clear interfaces, cohesive functions, and descriptive names, including physical units and mathematical meanings.
- Favor readability over cleverness; explain *why*, invariants, approximations, and non-obvious mathematics rather than narrating syntax.
- Minimize hidden state, magical dispatch, framework layers, speculative generalization, and unnecessary dependencies.
- Sophisticated algorithms are acceptable. Make their structure and assumptions understandable rather than replacing them with opaque simplifications.

## Architecture and proportion

- Separate mechanism from policy or judgment when it preserves user choice.
- Keep interfaces sufficiently clear that methods and components can be independently tested or replaced.
- Add automation, observability, configurability, and abstractions only when their benefits justify the associated cognitive cost.
- Prefer simple designs that satisfy the requirement; do not maximize simplicity at the expense of correctness or scientific validity.

## Evidence and scientific trust

- State the operating domain and limitations of calculations and models.
- Prefer independent analytical references, invariants, established benchmarks, and reproducible comparisons over tests that simply mirror implementation logic.
- Distinguish verified functionality from model validity and suitability for a particular application.
- A passing test is evidence about the case tested, not a blanket claim of correctness.

## LLM-assisted development

- LLMs can assist with design, coding, and critique; they must not become prerequisites for maintaining the result.
- Favor small changes preceded by substantive technical discussion. Humans decide when exploration becomes implementation.
- Preserve durable architectural reasoning outside ephemeral conversations.
- Do not add complexity merely because an LLM can generate or navigate it. The rate of code generation should not outpace the capacity for accountable review. 

## Tests for Conviviality

Could a competent engineer encountering the repository for the first time, without the original developer, conversation, or coding agent, understand its purpose, reconstruct major design choices, diagnose failures, modify behavior, and independently assess their changes?

Can a user understand what the software has done, recognize when its results should not be trusted, correct consequential inputs or choices, and take their work elsewhere without needing the developer?

Apply these test with judgment, not as a mandatory compliance checklist.
