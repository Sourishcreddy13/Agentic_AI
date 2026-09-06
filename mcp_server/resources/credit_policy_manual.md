# Synthetic Credit Policy Manual — Personal Loan Products
**Version:** 2025.1 | **Classification:** Internal Use Only (Synthetic Data)

---

## Section 1 — Eligibility

### 1.1 Minimum Income
Applicants must demonstrate a minimum declared annual income of INR 25,000.
Income below this threshold triggers specialist officer referral.
Applicants with income above INR 75,000 qualify for the Premium Lending Tier.

### 1.2 Age
Applicants must be 21–65 years of age (age at loan end-date must not exceed 65).

### 1.3 Residency and Employment
Domestic residents or long-term visa holders (12+ months remaining).
Salaried, self-employed, and registered business owners are eligible.
Gig/freelance applicants require 12 months of income history.

---

## Section 2 — KYC Requirements

### 2.1 Mandatory Checks
1. Identity Verification (photo ID, DOB match)
2. Watchlist/Sanctions/PEP Screening
3. Income Plausibility

### 2.2 KYC Outcomes
- **Pass**: All checks clear → proceed to credit assessment.
- **Manual Review**: Minor flag → compliance officer review.
- **Fail**: Watchlist hit or identity mismatch → application terminated immediately;
  no credit assessment performed.

---

## Section 3 — Credit Assessment

### 3.1 Debt-to-Income Ratio
- Standard cap: DTI ≤ 0.40
- Premium (income >75k): DTI ≤ 0.50
- Thin-file cap: DTI ≤ 0.30

### 3.2 Thin-File Definition
Bureau score < 500 OR fewer than 2 open accounts with payment history.
Thin-file applications must be referred to manual underwriting; automated
approval is not permitted.

### 3.3 Manual Underwriting Triggers
- Thin file
- DTI between threshold and 0.60
- KYC Manual Review result

---

## Section 4 — Pricing

### 4.1 APR by Score Band
| Score     | APR      | Max Principal |
|-----------|----------|---------------|
| 750–900   | 8.5%     | INR 5,000,000 |
| 700–749   | 9.5–10.5%| INR 1.5–3M    |
| 650–699   | 12.0%    | INR 750,000   |
| 600–649   | 13.5%    | INR 500,000   |
| <500      | Manual   | Per THIN-001  |

### 4.2 Offer Validity
All indicative offers valid for 30 days and subject to final underwriter sign-off.

---

## Section 5 — Offer Issuance

All offers are indicative. Final APR may vary ±0.5% from the representative rate.
Applications requesting principal above tier maximum require Level-2 Credit Officer sign-off.

---

*This document contains synthetic data only. No real applicant, bureau, or pricing
information is present. For training and evaluation purposes only.*
