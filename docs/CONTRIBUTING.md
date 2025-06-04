# Git Workflow Guidelines

These are the Git usage conventions for this repository.  
Please follow them whether you are a human developer or an AI agent like Claude.

---

## Branching Policy

- Main branch: `main`
    - Production-ready code only.
- Development branch: `dev`
    - Base for all feature branches.
- Feature branches:
    - Format: `feature/yyyyMMdd-short-description`
    - Example: `feature/20250604-claudecode-setup`
    - Always branch off from `dev`.

---

## Commit Message Convention

- Use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)
- Format: `type: short summary`
    - Types include: `feat`, `fix`, `docs`, `refactor`, `chore`, `test`
    - Example: `docs: create claudecode.yml`
- Use English for all commit messages.

---

## Pull Request Rules

- Pull requests must target the `dev` branch.
- Title should reflect the feature or fix being added.
- Include a short description in the PR body (1–3 sentences in English).
- Do **not** merge without a successful check.

---

## Other Conventions

- Avoid pushing directly to `main` or `dev`.
- Each pull request should be tied to a single purpose (no mixed commits).
- Cleanup commits (e.g., typo fixes) should be squashed before merging if possible.

---

Thank you for keeping this repository clean and consistent! 💻✨
