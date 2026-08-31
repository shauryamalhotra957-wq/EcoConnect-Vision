# Security policy

EcoConnect Vision processes camera frames and model artifacts locally.

- Do not commit webcam captures, private datasets, credentials, or model files derived from restricted data.
- Treat camera input and notebook outputs as sensitive until reviewed and deleted.
- Load only trusted Keras artifacts; record hashes for models used in evaluations.
- Do not present prototype classifications as compliance, enforcement, or safety decisions without human review.
- Review any future network or upload integration for consent, authentication, and retention.

Report suspected data exposure, unsafe model deserialization, or command injection privately to the repository owner with sanitized reproduction steps.
