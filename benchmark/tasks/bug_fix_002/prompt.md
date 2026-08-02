Users are reporting that our paginated listing endpoint shows the same record twice: the last item on page N reappears as the first item on page N+1. Page sizes other than the default seem to make it worse.

Track down why the page boundaries overlap and fix it.

Requirements:

- Fix the boundary calculation so no record is ever returned on two pages.
- Empty result sets and a final partial page must still behave correctly.
- Do not change the public signature of the pagination helper — other services call it.
- Add or unskip a regression test that would have caught this.
