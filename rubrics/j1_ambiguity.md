# J1: Ambiguity handling (failure mode F5)

Decide whether the agent handled ambiguity in the question correctly.

A question is **ambiguous** when a reasonable analyst could not answer it without choosing between
materially different readings, for example:
- **metric**: "best sellers" (by GMV, units, rating?), "revenue" (GMV or net revenue?) when the choice changes the answer;
- **entity**: a name that matches several objects (two products are called "Classic Panjabi"; two sellers are "Jamuna Fashion");
- **period**: "this year" vs fiscal year, "last quarter" when the context does not fix it.

PASS when:
- the question is ambiguous and the agent asked **one short, specific clarifying question** naming the options; or
- the question is ambiguous and the agent answered while **explicitly stating** the reading it chose and why
  (a reasonable, conventional default), so the user can correct it; or
- the question is clear and the agent answered without asking.

FAIL when:
- the question is ambiguous and the agent silently picked one reading;
- the agent asked a clarifying question about something that does not change the answer (needless clarification);
- the clarifying question is vague ("Can you clarify?") or asks several unrelated things.

Judge only ambiguity handling, not SQL correctness.
