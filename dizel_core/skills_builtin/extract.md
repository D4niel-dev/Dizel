---
name: extract
title: Extract structured data
description: Pull fields from messy text into clean JSON.
aliases: json
category: Work
---
Extract structured data from the text below.

- Use the schema if provided; otherwise propose a small schema first.
- Return valid JSON only, with null for missing values.
- Preserve exact names, dates, and numbers; do not guess.
- List any fields that were ambiguous after the JSON.
- Keep nested objects only when the source supports them.

{{input}}
