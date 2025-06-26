# Viola AI - A Proactive Decision Intelligence Dashboard

## 1. Project Overview & Strategic Rationale

This project is a prototype developed in response to the **AI-Readiness Pre-Screening Challenge**. The initial task was to analyze a lead generation tool. However, a foundational analysis revealed a critical limitation: lead generation tools are inherently **reactive**—they provide historical data on existing companies but lack the predictive power to inform high-stakes, forward-looking investment decisions.

> For a modern investment firm, simply knowing "who exists" is insufficient; the crucial advantage lies in knowing **"what comes next."**

Recognizing this gap, this project pivots from the initial concept to deliver a far more valuable solution: a **proactive Decision Intelligence Dashboard**. This tool is engineered not just to find leads, but to generate alpha by answering the three core questions an investment committee faces:

1.  **Which ventures will succeed?** (Predictive Analysis)
2.  **Which markets will grow?** (Trend Forecasting)
3.  **What are the hidden risks?** (Consultative AI)

This dashboard directly addresses the shortcomings of a standard lead generator by providing predictive, data-driven insights, transforming raw data into actionable intelligence.

---

## 2. Core Modules & Technical Implementation

The dashboard integrates three purpose-built modules, each designed to solve a specific problem.

| Module | Problem | Solution |
| :--- | :--- | :--- |
| **Startup & Company Predictor** | Evaluating a startup's future success is complex and subjective. A simple list of leads offers no insight into viability. | Employs a **Keras/TensorFlow-based neural network** to calculate success probability. Integrated with **SHAP (SHapley Additive exPlanations)** to provide auditable, transparent reasoning behind every prediction, detailing both positive and negative factors. |
| **Dynamic Market Forecaster** | Investment requires anticipating future trends, not just reacting to current ones. Lead generators provide no market momentum data. | An interactive tool leveraging the **Pytrends API** to fetch and visualize real-time Google Trends data. Users can dynamically compare keywords to identify emerging "waves," validate theses, and detect "fading" trends. The Flask backend generates charts on-the-fly with Matplotlib. |
| **AI Risk Management Consultant (Viola)** | Initial due diligence and risk assessment are time-consuming. An interactive consultant can streamline this preliminary analysis. | A specialized chatbot named **Viola**, powered by a **Google Vertex AI** model that has been **fine-tuned on a curated 10,000-token dataset** of risk management literature, financial case studies, and due diligence questionnaires, making it a highly specialized tool for initial risk assessment. |

---

## 3. Technology Stack

This project was built using a focused and robust set of technologies, with **Flask** serving as the sole backend framework.

| Category | Technology / Library | Purpose |
| :--- | :--- | :--- |
| **Backend Framework** | `Flask` | Routing, logic, and serving templates. |
| **Frontend** | `HTML5`, `Tailwind CSS` | Structure, interactivity, and modern UI/UX. |
| **Machine Learning** | `TensorFlow/Keras` | Building the predictive neural network. |
| | `Scikit-learn` | Data preprocessing and scaling. |
| | `SHAP` | Providing model explainability |
| | `Pandas`, `NumPy` | High-performance data manipulation. |
| **External APIs** | `Pytrends` | Real-time integration with Google Trends. |
| | `Google Cloud Vertex AI` | Serving the fine-tuned Generative AI model. |
| | `Matplotlib` | Dynamic, server-side chart generation. |