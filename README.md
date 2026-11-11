# SAD_patient_AI
# SAD AI Companion — Synthetic Patient Agent

## 1. Overview

This module provides a **synthetic Patient AI** for the SAD AI Companion project.

The Patient Agent uses **Qwen** to simulate fictional individuals experiencing winter-related low mood. Each patient is guided by a structured Intake Form containing their personal background, symptoms, cognitive patterns, and daily-life experiences.

The primary purpose is to support the development and evaluation of our **Support AI** through controlled, multi-turn conversations.

**Note:** All patient profiles are synthetic. They are intended for research and system testing, not clinical diagnosis or treatment.

## 2. Project Structure

```text
SAD_AI_Companion/
├── neutral_patient.py
├── generate_sad_intakes.py
├── .env.example
├── .gitignore
└── outputs/
    └── sad_intakes_neutral_candidates_20.jsonl
```

| File | Description |
|---|---|
| `neutral_patient.py` | Main Patient Agent module |
| `generate_sad_intakes.py` | Generates synthetic Intake Forms using Qwen |
| `sad_intakes_neutral_candidates_20.jsonl` | 20 generated patient profiles |
| `.env.example` | Example Qwen API configuration |
| `.gitignore` | Prevents local configuration and credentials from being committed |

## 3. Dataset Generation

The synthetic patient profiles are generated using a pipeline inspired by the **CACTUS** approach.

```text
PatternReframe Dataset
        ↓
Persona + Unhelpful Thought + Cognitive Pattern
        ↓
SAD-related Scenario Adaptation
        ↓
Qwen Intake Form Generation
        ↓
Synthetic Patient Profiles (JSONL)
```

### Data sources

**PatternReframe** provides fictional personas, unhelpful thoughts, and cognitive distortion labels.

Additional winter-related contexts are introduced to construct synthetic scenarios involving low motivation, reduced enjoyment, fatigue, social withdrawal, and other relevant experiences.

These adaptations are synthetic and should not be interpreted as clinically validated SAD cases.

### Intake Form structure

Each generated case contains a unique `case_id` and an `intake_form` with fields such as:

- `profile_name`
- `age`
- `personal_background`
- `presenting_concern`
- `winter_context`
- `symptoms`
- `duration_or_onset`
- `seasonal_recurrence`
- `daily_life_impact`
- `unhelpful_thought`
- `thought_pattern`
- `support_preference`
- `disclosure_style`
- `diagnosis_status`
- `initial_utterance`
- `additional_private_context`

The current dataset contains **20 generated candidate profiles**. These should undergo quality review before use in formal evaluation.

## 4. Patient AI Features

The current `NeutralPatient` implementation supports:

- **Intake-form-conditioned roleplay:** Qwen responds according to a selected patient profile.
- **Neutral conversational attitude:** Moderately engaged, somewhat reserved, and responsive to questions.
- **Multi-turn conversation:** Conversation history is maintained within each Patient Agent instance.
- **Gradual disclosure:** The patient is instructed to reveal private information progressively.
- **Cognitive-pattern consistency:** Responses are guided by the cognitive pattern specified in the Intake Form.
- **Basic over-compliance prevention:** The patient is instructed not to accept every suggestion or claim immediate recovery.
- **Multiple patient profiles:** Different patient cases can be loaded from the JSONL dataset.

The current module provides the Patient Agent interface. Automatic Support–Patient orchestration, persistent conversation logging, and comprehensive automated evaluation are not yet part of this core implementation.

## 5. Installation

### Requirements

- Python 3.10 or later
- Access to the configured Qwen inference service
- Required Python packages

Install dependencies:

```bash
pip install openai python-dotenv
```

### Environment configuration

Create a `.env` file in the project root using `.env.example` as a reference:

```dotenv
QWEN_BASE_URL=http://your-server:8000/v1
QWEN_MODEL=your-model-id
QWEN_API_KEY=your-api-key
```

Use the actual endpoint, model identifier, and credentials provided for the Qwen service.

**Security:** Do not commit `.env` or share real API credentials in the repository.

The current implementation uses the OpenAI Python client and expects an OpenAI-compatible chat-completions endpoint.

It also requests `enable_thinking=False` through `chat_template_kwargs`, which requires a compatible Qwen deployment.

## 6. Running the Patient Agent

Run:

```bash
python neutral_patient.py
```

The current standalone script will:

1. Load the 20 synthetic patient profiles.
2. Randomly select one case.
3. Initialize a `NeutralPatient` instance.
4. Display the selected patient's opening statement.
5. Send one predefined Support message to Qwen.
6. Print the generated patient response.

**Important:** The standalone script currently tests one response. The `NeutralPatient` class itself supports repeated calls for multi-turn conversations.

## 7. Integration with Support AI

The Patient Agent is designed to communicate with the Support AI through two methods:

| Method | Description |
|---|---|
| `get_opening()` | Returns the patient's initial message |
| `respond(support_message)` | Generates the next patient reply based on the Support AI message and existing conversation history |

### Example integration

```python
from neutral_patient import NeutralPatient, load_patients

# Load synthetic patient profiles
cases = load_patients(
    "outputs/sad_intakes_neutral_candidates_20.jsonl"
)

# Select a patient
patient = NeutralPatient(cases[0])

# Patient starts the conversation
patient_message = patient.get_opening()

# Example multi-turn interaction
for turn in range(5):

    # Replace with the actual Support AI interface
    support_message = support_ai.respond(patient_message)

    # Generate next patient response
    patient_message = patient.respond(support_message)
```

**Note:** `support_ai.respond()` is a placeholder representing the teammate's Support AI interface. It must be replaced with the actual function or API call.

### Integration workflow

```text
Select Patient Intake Form
          ↓
Initialize NeutralPatient
          ↓
Patient Initial Utterance
          ↓
       Support AI
          ↓
     Support Response
          ↓
       Patient AI
          ↓
      Patient Reply
          ↓
    Repeat for N Turns
```

### Important integration rules

1. Select **one patient profile per conversation session**.
2. Create a **new `NeutralPatient` instance** for each independent session.
3. Only send the patient's generated messages to the Support AI.
4. Do **not** expose the private Intake Form, hidden context, or cognitive-pattern labels to the Support AI.
5. Maintain separate conversation histories for different sessions.
6. Handle Qwen API failures and preserve conversation records during integration testing.

## 8. Current Development Status

| Component | Status |
|---|---|
| PatternReframe data exploration | Completed |
| Initial SAD scenario adaptation | Completed |
| Qwen Intake Form generation | Implemented |
| 20 synthetic patient profiles | Generated |
| Neutral Patient Agent | Implemented |
| Multi-turn conversational memory | Implemented |
| Support AI integration | Pending |
| Automatic conversation logging | Pending |
| Dataset-wide quality validation | Pending |
| Positive / Negative attitude refinement | Future work |
| Automated evaluation framework | Future work |

## 9. Research Considerations

The Patient Agent is intended to provide a controlled environment for examining Support AI behavior.

Potential evaluation dimensions include:

- Persona consistency
- Cognitive-pattern consistency
- Naturalness of dialogue
- Gradual disclosure
- Resistance to over-compliance
- Seasonal-context plausibility
- Response consistency across multiple turns

Synthetic patient responses should **not** be treated as evidence of therapeutic effectiveness or clinical safety.

Future experiments may extend the Neutral Agent to Positive and Negative conversational attitudes and evaluate behavioral differences under controlled conditions.

## 10. References

- **CACTUS:** *Towards Psychological Counseling Conversations using Cognitive Behavioral Theory*  
  https://github.com/YCY047/cactus

- **PatternReframe:** *Training Models to Generate, Recognize, and Reframe Unhelpful Thoughts*  
  Dataset accessed through the ParlAI `reframe_thoughts` task.

---

**Current integration goal:** Enable the Support AI to interact with different synthetic patients through the `NeutralPatient` interface, maintain multi-turn conversations, and collect outputs for subsequent evaluation.
