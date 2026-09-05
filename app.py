from flask import Flask, render_template, request, jsonify
import ollama
from PyPDF2 import PdfReader

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/generate-question", methods=["POST"])
def generate_question():
    job_description = request.form.get("job_description", "")
    resume = request.files.get("resume")
    question_number = request.form.get("question_number", "1")
    previous_question = request.form.get("previous_question", "")
    previous_answer = request.form.get("previous_answer", "")
    previous_feedback = request.form.get("previous_feedback", "")

    if not job_description:
        return jsonify({"error": "Job description is required."}), 400

    if not resume:
        return jsonify({"error": "Resume is required."}), 400

    try:
        reader = PdfReader(resume)

        resume_text = ""

        for page in reader.pages:
            text = page.extract_text()

            if text:
                resume_text += text + "\n"

    except Exception as e:
        return jsonify({
            "error": "Could not read the resume PDF: " + str(e)
        }), 500

    if not resume_text.strip():
        return jsonify({
            "error": "Could not extract text from the resume. "
                     "Make sure the PDF contains selectable text."
        }), 400

    if question_number == "1":

        prompt = f"""
You are a professional job interviewer.

The candidate is preparing for a job interview.

JOB DESCRIPTION:
{job_description}

CANDIDATE RESUME:
{resume_text}

Generate Question 1 of a realistic mock interview.

The question must be based on BOTH:
1. The job description
2. The candidate's resume

Make the question personalized.

Prefer a behavioral or experience-based question for the first question.

Do not provide the answer.

Return ONLY the interview question.
"""

    else:

        prompt = f"""
You are a professional interviewer conducting a realistic mock interview.

JOB DESCRIPTION:
{job_description}

CANDIDATE RESUME:
{resume_text}

PREVIOUS QUESTION:
{previous_question}

CANDIDATE'S PREVIOUS ANSWER:
{previous_answer}

PREVIOUS AI FEEDBACK:
{previous_feedback}

This is Question {question_number} of the interview.

Generate the next interview question.

The question should:
- Be relevant to the job description
- Be personalized using the candidate's resume
- Build naturally from the previous answer when appropriate
- Test a different skill or competency from the previous question
- Feel like a real professional interview

For Question 2:
Prefer a role-specific or technical question when the job allows it.

For Question 3:
Prefer a behavioral, problem-solving, situational, or deeper follow-up question.

Do not repeat the previous question.

Do not provide the answer.

Return ONLY the interview question.
"""

    try:
        response = ollama.chat(
            model="qwen2.5:3b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        question = response["message"]["content"].strip()

        return jsonify({
            "question": question,
            "question_number": question_number
        })

    except Exception as e:
        return jsonify({
            "error": "Ollama error: " + str(e)
        }), 500


@app.route("/evaluate-answer", methods=["POST"])
def evaluate_answer():

    data = request.get_json()

    question = data.get("question", "")
    answer = data.get("answer", "")

    if not question:
        return jsonify({
            "error": "Interview question is required."
        }), 400

    if not answer:
        return jsonify({
            "error": "Candidate answer is required."
        }), 400

    prompt = f"""
You are an expert professional interview evaluator.

Evaluate the candidate's answer to the interview question.

INTERVIEW QUESTION:
{question}

CANDIDATE ANSWER:
{answer}

Evaluate the answer using these four categories.

1. RELEVANCE
Score from 0 to 10.

Consider:
- Does the candidate directly answer the question?
- Is the answer relevant?
- Does the candidate stay focused?

2. CLARITY
Score from 0 to 10.

Consider:
- Is the answer easy to understand?
- Is it logically organized?
- Does the candidate communicate clearly?

3. STAR STRUCTURE
Score from 0 to 10.

Evaluate:

S = Situation
T = Task
A = Action
R = Result

Check whether the candidate explains:
- What situation they faced
- What task or responsibility they had
- What actions THEY personally took
- What result was achieved

For technical questions, do not penalize the candidate simply because
STAR is not naturally appropriate. Evaluate whether the answer is
well structured for that type of question.

4. OVERALL
Give an overall score from 0 to 10 based on the complete quality
of the answer.

Use EXACTLY this format:

Relevance: X/10
Clarity: X/10
STAR Structure: X/10
Overall: X/10

Strengths:
- Write one specific strength.
- Write another specific strength if applicable.

Areas to Improve:
- Write one specific improvement.
- Write another specific improvement if applicable.

Suggested Improvement:
Explain briefly how the candidate could improve the answer.

Be constructive, professional, and specific.

Do not give an unrelated answer.
"""

    try:

        response = ollama.chat(
            model="qwen2.5:3b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        evaluation = response["message"]["content"].strip()

        return jsonify({
            "evaluation": evaluation
        })

    except Exception as e:

        return jsonify({
            "error": "Ollama evaluation error: " + str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)