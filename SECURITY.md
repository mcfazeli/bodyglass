# Security policy

BodyGlass handles people's own medical images, so privacy and security reports are taken seriously.

## Reporting a vulnerability

Please report security issues privately, not in a public issue. Use GitHub's private vulnerability reporting
(the **Security** tab of this repository, then **Report a vulnerability**). Include the steps to reproduce,
the version or commit, and the impact you expect.

Reports of particular interest:

- anything that lets a web page or another device read data from the local server (`127.0.0.1:8765`)
- anything that sends scan data, chats or tokens off the Mac without the user choosing it
- weaknesses in the ChatGPT sign-in, token storage or model download checks

## Supported versions

Only the latest version on `main` receives fixes.
