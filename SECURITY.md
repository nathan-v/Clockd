# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| latest  | Yes       |

## Reporting a Vulnerability

Clockd processes video uploads and interacts with camera systems, so security is taken seriously.

If you discover a security vulnerability, please report it responsibly. For issues that do not require private disclosure (most cases given this is a locally-run service), please open a GitHub issue.

For sensitive security issues that require private disclosure, please [contact](https://www.nathanv.com/contact) the maintainer directly.

Please include:
- A description of the vulnerability
- Steps to reproduce the issue
- Any potential impact
- Suggested fix (if you have one)

You should receive a response within 72 hours. Security fixes will be prioritized and released as soon as possible.

## Known Limitations

- **No authentication**: Clockd does not currently implement API authentication. It is designed to run on a trusted local network. Do not expose it directly to the internet without a reverse proxy that handles authentication.
- **No rate limiting**: There is no built-in rate limiting. Use a reverse proxy if you need this.
