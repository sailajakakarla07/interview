import os
import json
import html

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from groq import Groq
from dotenv import load_dotenv

from audio_recorder_streamlit import audio_recorder

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER

import ai_engine

from database import (
    create_tables,
    create_session,
    save_answer,
    update_session_score,
    get_sessions,
    get_session_metrics,
    get_session_answers
)

from resume_parser import (
    extract_pdf_text,
    extract_docx_text
)


# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is missing in your .env file.")
    st.stop()

groq_client = Groq(api_key=GROQ_API_KEY)

create_tables()

st.set_page_config(
    page_title="AI Interview Bot",
    page_icon="🤖",
    layout="wide"
)


# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------

defaults = {
    "started": False,
    "finished": False,
    "questions": [],
    "question_index": 0,
    "session_id": None,
    "job_title": "",
    "job_description": "",
    "resume": "",
    "scores": [],
    "clarity_scores": [],
    "relevance_scores": [],
    "structure_scores": [],
    "feedback": [],
    "improvement_tips": [],
    "answers_data": [],
    "current_evaluation": None,
    "spoken_text": "",
    "answer_text": "",
    "current_question_is_followup": False,
    "final_summary": ""
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# -------------------------------------------------
# SPEECH TO TEXT
# -------------------------------------------------

def speech_to_text(audio_bytes):

    try:

        os.makedirs("uploads", exist_ok=True)

        audio_path = "uploads/answer.wav"

        with open(audio_path, "wb") as file:
            file.write(audio_bytes)

        with open(audio_path, "rb") as audio_file:

            transcription = groq_client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3-turbo",
                language="en",
                response_format="json",
                temperature=0.0
            )

        return transcription.text

    except Exception as e:

        st.error("Unable to convert speech to text.")
        st.error(str(e))

        return ""


# -------------------------------------------------
# TEXT TO SPEECH
# -------------------------------------------------

def speak_question(question):

    safe_question = json.dumps(question)

    components.html(
        f"""
        <script>

        const text = {safe_question};

        window.speechSynthesis.cancel();

        const speech =
            new SpeechSynthesisUtterance(text);

        speech.rate = 0.95;
        speech.pitch = 1;
        speech.volume = 1;

        window.speechSynthesis.speak(speech);

        </script>
        """,
        height=0
    )


# -------------------------------------------------
# QUESTION HELPERS
# -------------------------------------------------

def clean_questions(questions):

    cleaned = []

    if not questions:
        return cleaned

    for question in questions:

        if isinstance(question, dict):

            question = question.get(
                "question",
                question.get("text", "")
            )

        question = str(question).strip()

        if question:
            cleaned.append(question)

    return cleaned[:5]


def get_score(evaluation, key):

    try:
        return float(evaluation.get(key, 0))

    except Exception:
        return 0.0


# -------------------------------------------------
# RESET
# -------------------------------------------------

def reset_interview():

    st.session_state.started = False
    st.session_state.finished = False
    st.session_state.questions = []
    st.session_state.question_index = 0
    st.session_state.session_id = None

    st.session_state.job_title = ""
    st.session_state.job_description = ""
    st.session_state.resume = ""

    st.session_state.scores = []
    st.session_state.clarity_scores = []
    st.session_state.relevance_scores = []
    st.session_state.structure_scores = []

    st.session_state.feedback = []
    st.session_state.improvement_tips = []
    st.session_state.answers_data = []

    st.session_state.current_evaluation = None

    st.session_state.spoken_text = ""
    st.session_state.answer_text = ""

    st.session_state.current_question_is_followup = False

    st.session_state.final_summary = ""


# -------------------------------------------------
# PDF REPORT
# -------------------------------------------------

def create_pdf_report():

    os.makedirs("uploads", exist_ok=True)

    pdf_path = "uploads/interview_report.pdf"

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = styles["Title"]
    title_style.alignment = TA_CENTER

    normal_style = styles["BodyText"]

    story = []

    story.append(
        Paragraph(
            "AI INTERVIEW REPORT",
            title_style
        )
    )

    story.append(Spacer(1, 15))

    overall_score = (
        sum(st.session_state.scores)
        / len(st.session_state.scores)
        if st.session_state.scores
        else 0
    )

    story.append(
        Paragraph(
            f"<b>Job Title:</b> "
            f"{html.escape(st.session_state.job_title)}",
            normal_style
        )
    )

    story.append(Spacer(1, 8))

    story.append(
        Paragraph(
            f"<b>Overall Score:</b> "
            f"{overall_score:.1f}/100",
            normal_style
        )
    )

    story.append(Spacer(1, 15))

    data = [
        [
            "Question",
            "Clarity",
            "Relevance",
            "Structure",
            "Total"
        ]
    ]

    for i, item in enumerate(
        st.session_state.answers_data,
        start=1
    ):

        data.append([
            f"Q{i}",
            f"{item['clarity']:.1f}",
            f"{item['relevance']:.1f}",
            f"{item['structure']:.1f}",
            f"{item['total']:.1f}"
        ])

    table = Table(data)

    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (1, 1), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
        ])
    )

    story.append(table)

    story.append(Spacer(1, 20))

    story.append(
        Paragraph(
            "<b>Interview Summary</b>",
            styles["Heading2"]
        )
    )

    summary_text = (
        st.session_state.final_summary
        or "No summary available."
    )

    summary_text = html.escape(summary_text)
    summary_text = summary_text.replace("\n", "<br/>")

    story.append(
        Paragraph(
            summary_text,
            normal_style
        )
    )

    story.append(Spacer(1, 20))

    story.append(
        Paragraph(
            "<b>Answer Feedback</b>",
            styles["Heading2"]
        )
    )

    for i, item in enumerate(
        st.session_state.answers_data,
        start=1
    ):

        story.append(
            Paragraph(
                f"<b>Question {i}</b>",
                normal_style
            )
        )

        feedback = html.escape(
            item["feedback"]
        )

        tip = html.escape(
            item["improvement_tip"]
        )

        story.append(
            Paragraph(
                f"<b>Feedback:</b> {feedback}",
                normal_style
            )
        )

        story.append(
            Paragraph(
                f"<b>Improvement Tip:</b> {tip}",
                normal_style
            )
        )

        story.append(Spacer(1, 10))

    doc.build(story)

    return pdf_path


# -------------------------------------------------
# TITLE
# -------------------------------------------------

st.title("🤖 AI Interview Bot")

st.write(
    "Practice your interview with an AI interviewer "
    "powered by Groq."
)

st.divider()


# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

page = st.sidebar.radio(
    "Navigation",
    [
        "💼 Interview",
        "📊 Session History",
        "📈 Progress Dashboard"
    ]
)


# =================================================
# PROGRESS DASHBOARD
# =================================================

if page == "📈 Progress Dashboard":

    st.header("📈 Progress Dashboard")

    metrics = get_session_metrics()

    if not metrics:

        st.info(
            "Complete at least one interview "
            "session to see your progress."
        )

    else:

        rows = []

        for row in metrics:

            (
                session_id,
                job_title,
                overall,
                created_at,
                clarity,
                relevance,
                structure
            ) = row

            rows.append({
                "Session": session_id,
                "Job": job_title,
                "Overall Score": float(overall or 0),
                "Clarity": float(clarity or 0),
                "Relevance": float(relevance or 0),
                "Structure": float(structure or 0),
                "Date": str(created_at)[:10]
            })

        df = pd.DataFrame(rows)

        st.subheader("Overall Score Trend")

        overall_df = df[
            ["Session", "Overall Score"]
        ].set_index("Session")

        st.line_chart(overall_df)

        st.subheader("Interview Skill Trends")

        skill_df = df[
            [
                "Session",
                "Clarity",
                "Relevance",
                "Structure"
            ]
        ].set_index("Session")

        st.line_chart(skill_df)

        st.subheader("Session Details")

        display_df = df.copy()

        display_df["Overall Score"] = \
            display_df["Overall Score"].round(1)

        display_df["Clarity"] = \
            display_df["Clarity"].round(1)

        display_df["Relevance"] = \
            display_df["Relevance"].round(1)

        display_df["Structure"] = \
            display_df["Structure"].round(1)

        st.dataframe(
            display_df,
            use_container_width=True
        )

        if len(df) >= 2:

            first_score = \
                df.iloc[0]["Overall Score"]

            latest_score = \
                df.iloc[-1]["Overall Score"]

            improvement = \
                latest_score - first_score

            if improvement > 0:

                st.success(
                    f"🎉 Your score improved by "
                    f"{improvement:.1f} points."
                )

            elif improvement < 0:

                st.warning(
                    f"Your score decreased by "
                    f"{abs(improvement):.1f} points."
                )

            else:

                st.info(
                    "Your score is currently stable."
                )


# =================================================
# SESSION HISTORY
# =================================================

elif page == "📊 Session History":

    st.header("📊 Interview Session History")

    sessions = get_sessions()

    if not sessions:

        st.info(
            "No interview sessions found."
        )

    else:

        for session in sessions:

            session_id = session[0]
            job_title = session[1]
            score = float(session[2] or 0)
            date = session[3]

            st.subheader(
                f"💼 {job_title}"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Overall Score",
                    f"{score:.1f}/100"
                )

            with col2:

                st.write(
                    f"Session ID: {session_id}"
                )

            with col3:

                st.write(
                    str(date)[:19]
                )

            st.divider()


# =================================================
# INTERVIEW PAGE
# =================================================

else:

    # -------------------------------------------------
    # INTERVIEW SETUP
    # -------------------------------------------------

    if not st.session_state.started:

        st.header("💼 Interview Setup")

        job_title = st.text_input(
            "Job Title",
            placeholder=(
                "Example: Junior Java Developer"
            )
        )

        job_description = st.text_area(
            "Job Description",
            placeholder=(
                "Paste the job description here..."
            ),
            height=180
        )

        st.subheader("📄 Resume")

        uploaded_resume = st.file_uploader(
            "Upload Resume",
            type=["pdf", "docx"]
        )

        pasted_resume = st.text_area(
            "Or paste your resume",
            placeholder=(
                "Paste your resume text here..."
            ),
            height=220
        )

        resume = pasted_resume

        if uploaded_resume is not None:

            try:

                os.makedirs("uploads", exist_ok=True)

                file_path = os.path.join(
                    "uploads",
                    uploaded_resume.name
                )

                with open(file_path, "wb") as file:

                    file.write(
                        uploaded_resume.getbuffer()
                    )

                if uploaded_resume.name.lower().endswith(".pdf"):

                    resume = extract_pdf_text(
                        file_path
                    )

                elif uploaded_resume.name.lower().endswith(".docx"):

                    resume = extract_docx_text(
                        file_path
                    )

                st.success(
                    "✅ Resume uploaded successfully."
                )

            except Exception as e:

                st.error(
                    "Unable to read the resume."
                )

                st.exception(e)

        st.divider()

        if st.button(
            "🚀 Start Interview",
            use_container_width=True
        ):

            if not job_title.strip():

                st.warning(
                    "Please enter a job title."
                )

                st.stop()

            if not job_description.strip():

                st.warning(
                    "Please enter a job description."
                )

                st.stop()

            if not resume.strip():

                st.warning(
                    "Please upload or paste your resume."
                )

                st.stop()

            with st.spinner(
                "🤖 Generating personalized interview questions..."
            ):

                try:

                    questions = ai_engine.generate_questions(
                        job_title,
                        job_description,
                        resume
                    )

                    questions = clean_questions(
                        questions
                    )

                    if not questions:

                        st.error(
                            "No questions generated."
                        )

                        st.stop()

                    session_id = create_session(
                        job_title,
                        job_description,
                        resume
                    )

                    st.session_state.job_title = \
                        job_title

                    st.session_state.job_description = \
                        job_description

                    st.session_state.resume = \
                        resume

                    st.session_state.questions = \
                        questions

                    st.session_state.question_index = 0

                    st.session_state.session_id = \
                        session_id

                    st.session_state.scores = []

                    st.session_state.clarity_scores = []

                    st.session_state.relevance_scores = []

                    st.session_state.structure_scores = []

                    st.session_state.feedback = []

                    st.session_state.improvement_tips = []

                    st.session_state.answers_data = []

                    st.session_state.current_evaluation = None

                    st.session_state.spoken_text = ""

                    st.session_state.answer_text = ""

                    st.session_state.started = True

                    st.session_state.finished = False

                    st.session_state.current_question_is_followup = False

                    st.rerun()

                except Exception as e:

                    st.error(
                        "Unable to start interview."
                    )

                    st.exception(e)


    # -------------------------------------------------
    # ACTIVE INTERVIEW
    # -------------------------------------------------

    if (
        st.session_state.started
        and not st.session_state.finished
    ):

        questions = st.session_state.questions

        index = st.session_state.question_index

        total_questions = len(questions)

        if index < total_questions:

            question = questions[index]

            st.header(
                f"Question {index + 1}"
                f" of {total_questions}"
            )

            st.progress(
                (index + 1) / total_questions
            )

            if st.session_state.current_question_is_followup:

                st.warning(
                    "🧠 Adaptive Follow-up Question"
                )

            st.info(
                f"🤖 **Interviewer:**\n\n{question}"
            )

            # -------------------------------
            # TEXT TO SPEECH
            # -------------------------------

            speak_question(question)

            st.divider()

            # -------------------------------
            # ANSWER METHOD
            # -------------------------------

            st.subheader("📝 Choose Your Answer Method")

            answer_method = st.radio(
                "Select one option:",
                ["⌨️ Type Answer", "🎙️ Voice Answer"],
                horizontal=True,
                key=f"answer_method_{index}"
            )

            answer = ""

            # -------------------------------
            # TYPE ANSWER
            # -------------------------------

            if answer_method == "⌨️ Type Answer":

                st.subheader("⌨️ Type Your Answer")

                answer = st.text_area(
                    "Your Answer",
                    placeholder="Type your answer here...",
                    height=220,
                    key=f"answer_box_{index}"
                )

                st.session_state.answer_text = answer
                st.session_state.spoken_text = ""

            # -------------------------------
            # VOICE ANSWER
            # -------------------------------

            else:

                st.subheader("🎙️ Speak Your Answer")

                st.write(
                    "Click the microphone, speak your answer, "
                    "and stop recording when you finish."
                )

                audio_bytes = audio_recorder(
                    text="🎙️ Start Recording",
                    recording_color="#ff4b4b",
                    neutral_color="#0068c9",
                    icon_name="microphone",
                    icon_size="2x"
                )

                if audio_bytes:

                    with st.spinner(
                        "🎧 Converting your speech to text..."
                    ):

                        spoken_text = speech_to_text(
                            audio_bytes
                        )

                    if spoken_text:

                        st.session_state.spoken_text = spoken_text
                        st.session_state.answer_text = spoken_text

                        st.success(
                            "✅ Voice recorded and converted to text!"
                        )

                        st.write("**Recognized Answer:**")
                        st.info(spoken_text)

                        answer = spoken_text

                else:
                    answer = st.session_state.spoken_text

                st.session_state.answer_text = answer

            st.divider()

            # -------------------------------
            # SUBMIT
            # -------------------------------

            if st.button(
                "📤 Submit Answer",
                use_container_width=True,
                key=f"submit_{index}"
            ):

                if not answer.strip():

                    st.warning(
                        "Please provide an answer."
                    )

                    st.stop()

                with st.spinner(
                    "🤖 AI is evaluating your answer..."
                ):

                    try:

                        evaluation = \
                            ai_engine.evaluate_answer(
                                question,
                                answer,
                                st.session_state.job_description,
                                st.session_state.resume
                            )

                    except Exception as e:

                        st.error(
                            "Unable to evaluate answer."
                        )

                        st.exception(e)

                        st.stop()

                clarity = \
                    get_score(
                        evaluation,
                        "clarity"
                    )

                relevance = \
                    get_score(
                        evaluation,
                        "relevance"
                    )

                structure = \
                    get_score(
                        evaluation,
                        "structure"
                    )

                total = \
                    clarity + relevance + structure

                feedback = \
                    evaluation.get(
                        "feedback",
                        "No feedback available."
                    )

                improvement_tip = \
                    evaluation.get(
                        "improvement_tip",
                        "Keep practicing."
                    )

                st.session_state.scores.append(
                    total
                )

                st.session_state.clarity_scores.append(
                    clarity
                )

                st.session_state.relevance_scores.append(
                    relevance
                )

                st.session_state.structure_scores.append(
                    structure
                )

                st.session_state.feedback.append(
                    feedback
                )

                st.session_state.improvement_tips.append(
                    improvement_tip
                )

                st.session_state.answers_data.append({
                    "question": question,
                    "answer": answer,
                    "clarity": clarity,
                    "relevance": relevance,
                    "structure": structure,
                    "total": total,
                    "feedback": feedback,
                    "improvement_tip": improvement_tip
                })

                save_answer(
                    st.session_state.session_id,
                    question,
                    answer,
                    clarity,
                    relevance,
                    structure,
                    total,
                    feedback,
                    improvement_tip
                )

                st.session_state.current_evaluation = {
                    "clarity": clarity,
                    "relevance": relevance,
                    "structure": structure,
                    "total": total,
                    "feedback": feedback,
                    "improvement_tip": improvement_tip
                }

                st.rerun()


            # -------------------------------------------------
            # EVALUATION DISPLAY
            # -------------------------------------------------

            evaluation = \
                st.session_state.current_evaluation

            if evaluation is not None:

                st.divider()

                st.header(
                    "📊 Answer Evaluation"
                )

                col1, col2, col3, col4 = \
                    st.columns(4)

                with col1:

                    st.metric(
                        "Clarity",
                        f"{evaluation['clarity']:.1f}/30"
                    )

                with col2:

                    st.metric(
                        "Relevance",
                        f"{evaluation['relevance']:.1f}/35"
                    )

                with col3:

                    st.metric(
                        "Structure / STAR",
                        f"{evaluation['structure']:.1f}/35"
                    )

                with col4:

                    st.metric(
                        "Total",
                        f"{evaluation['total']:.1f}/100"
                    )

                st.subheader(
                    "💬 Feedback"
                )

                st.write(
                    evaluation["feedback"]
                )

                st.subheader(
                    "🎯 Improvement Tip"
                )

                st.write(
                    evaluation["improvement_tip"]
                )

                st.divider()

                # -------------------------------------------------
                # ADAPTIVE FOLLOW UP
                # -------------------------------------------------

                if (
                    index + 1 < total_questions
                ):

                    if st.button(
                        "🧠 Generate Adaptive Follow-up",
                        use_container_width=True
                    ):

                        with st.spinner(
                            "🤖 Generating adaptive question..."
                        ):

                            try:

                                followup = \
                                    ai_engine.generate_follow_up_question(
                                        question,
                                        st.session_state.answer_text,
                                        evaluation,
                                        st.session_state.job_description,
                                        st.session_state.resume
                                    )

                                st.session_state.questions.insert(
                                    index + 1,
                                    followup
                                )

                                st.session_state.question_index += 1

                                st.session_state.current_evaluation = None

                                st.session_state.spoken_text = ""

                                st.session_state.answer_text = ""

                                st.session_state.current_question_is_followup = True

                                st.rerun()

                            except Exception as e:

                                st.error(
                                    "Unable to generate follow-up."
                                )

                                st.exception(e)

                    if st.button(
                        "➡️ Next Question",
                        use_container_width=True
                    ):

                        st.session_state.question_index += 1

                        st.session_state.current_evaluation = None

                        st.session_state.spoken_text = ""

                        st.session_state.answer_text = ""

                        st.session_state.current_question_is_followup = False

                        st.rerun()

                else:

                    if st.button(
                        "🏁 Finish Interview",
                        use_container_width=True
                    ):

                        overall_score = (
                            sum(
                                st.session_state.scores
                            )
                            / len(
                                st.session_state.scores
                            )
                            if st.session_state.scores
                            else 0
                        )

                        update_session_score(
                            st.session_state.session_id,
                            overall_score
                        )

                        with st.spinner(
                            "🤖 Preparing your final report..."
                        ):

                            try:

                                st.session_state.final_summary = \
                                    ai_engine.generate_final_summary(
                                        st.session_state.scores,
                                        st.session_state.feedback
                                    )

                            except Exception:

                                st.session_state.final_summary = \
                                    "Interview completed successfully."

                        st.session_state.finished = True

                        st.session_state.current_evaluation = None

                        st.rerun()


    # -------------------------------------------------
    # FINISHED INTERVIEW
    # -------------------------------------------------

    if st.session_state.finished:

        st.header(
            "🎉 Interview Completed!"
        )

        scores = st.session_state.scores

        overall_score = (
            sum(scores) / len(scores)
            if scores
            else 0
        )

        st.metric(
            "🏆 Overall Interview Score",
            f"{overall_score:.1f}/100"
        )

        st.divider()

        # ---------------------------------------------
        # FINAL SUMMARY
        # ---------------------------------------------

        st.subheader(
            "📝 Final Interview Summary"
        )

        st.write(
            st.session_state.final_summary
        )

        st.divider()

        # ---------------------------------------------
        # QUESTION SCORES
        # ---------------------------------------------

        st.subheader(
            "📊 Question-wise Scores"
        )

        for i, score in enumerate(
            scores,
            start=1
        ):

            st.write(
                f"Question {i}: "
                f"**{score:.1f}/100**"
            )

            st.progress(
                min(
                    max(score / 100, 0),
                    1
                )
            )

        st.divider()

        # ---------------------------------------------
        # FEEDBACK
        # ---------------------------------------------

        st.subheader(
            "💡 Interview Feedback"
        )

        for i, feedback in enumerate(
            st.session_state.feedback,
            start=1
        ):

            st.write(
                f"**Question {i}:** "
                f"{feedback}"
            )

            st.write(
                f"🎯 **Tip:** "
                f"{st.session_state.improvement_tips[i - 1]}"
            )

        st.divider()

        # ---------------------------------------------
        # PDF DOWNLOAD
        # ---------------------------------------------

        st.subheader(
            "📄 Download Report"
        )

        try:

            pdf_path = create_pdf_report()

            with open(
                pdf_path,
                "rb"
            ) as pdf_file:

                st.download_button(
                    label="⬇️ Download Interview Report PDF",
                    data=pdf_file,
                    file_name="AI_Interview_Report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

        except Exception as e:

            st.error(
                "Unable to create PDF report."
            )

            st.exception(e)

        st.divider()

        st.success(
            "✅ Your interview session has been saved."
        )

        if st.button(
            "🔄 Start New Interview",
            use_container_width=True
        ):

            reset_interview()

            st.rerun()
