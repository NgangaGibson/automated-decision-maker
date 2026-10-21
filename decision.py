import re
import os
import json
from typing import List, Optional, Literal
from openai import OpenAI
from pydantic import BaseModel, Field

# Initialize OpenAI Client (Set OPENAI_API_KEY in your environment)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ==========================================
# 1. STRUCTURED OUTPUT MODELS (Pydantic)
# ==========================================

class DomainClassification(BaseModel):
    field_of_study: str = Field(description="The specific field of study (e.g., Medicine, Finance, Law, Software Engineering, Education, General)")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0")
    reasoning: str = Field(description="Brief explanation of why this domain was chosen")

class UniversalDecision(BaseModel):
    domain: str = Field(description="The field of study applied")
    decision_type: str = Field(description="e.g., 'Approval', 'Rejection', 'Recommendation', 'Escalation', 'Information'")
    action_required: str = Field(description="The exact, actionable step to take")
    reasoning: str = Field(description="Step-by-step logical deduction based on domain principles")
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = Field(description="Assessment of risk if this decision is executed")
    human_in_the_loop_required: bool = Field(description="True if this decision requires human expert validation before execution")
    citations_or_principles: List[str] = Field(description="Key principles, laws, or best practices referenced")

# ==========================================
# 2. DOMAIN REGISTRY (Extensible to ALL fields)
# ==========================================

DOMAIN_REGISTRY = {
    "Medicine/Healthcare": "Focus on patient safety, triage, evidence-based guidelines (e.g., WHO, CDC). NEVER prescribe medication. Always recommend professional medical consultation for high-risk symptoms.",
    "Finance/Investing": "Focus on risk tolerance, regulatory compliance (SEC, fiduciary duty), and market principles. Never guarantee returns. Flag high-risk or speculative requests.",
    "Law/Legal": "Focus on jurisdiction, liability, and general legal principles. ALWAYS include a disclaimer that this is not formal legal advice and requires a licensed attorney.",
    "Software Engineering": "Focus on scalability, security (OWASP), maintainability, and best practices. Recommend specific architectures, libraries, or debugging steps.",
    "Education/Academia": "Focus on pedagogical value, academic integrity, and structured learning paths. Adapt complexity to the user's stated level.",
    "Business/Management": "Focus on ROI, stakeholder impact, operational efficiency, and strategic alignment.",
    "General/Other": "Apply common sense, ethical AI principles, and general knowledge. Escalate if the request is ambiguous or potentially harmful."
}

# ==========================================
# 3. THE UNIVERSAL DECISION ENGINE
# ==========================================

class UniversalDecisionMaker:
    def __init__(self):
        self.router_prompt = """
        You are an expert domain classifier. Analyze the user's request and determine the most relevant field of study.
        Available fields: {fields}
        Return ONLY valid JSON matching the DomainClassification schema.
        """
        
        self.expert_prompt = """
        You are a world-class expert in {domain}. 
        Domain Guidelines: {guidelines}
        
        Analyze the user's request and make a structured, logical decision. 
        Prioritize safety, ethics, and established best practices for this field.
        Return ONLY valid JSON matching the UniversalDecision schema.
        """

    def _classify_domain(self, user_request: str) -> DomainClassification:
        """Step 1: Route the request to the correct field of study."""
        fields_list = ", ".join(DOMAIN_REGISTRY.keys())
        
        response = client.beta.chat.completions.parse(
            model="gpt-4o", # or gpt-4o-mini for cost efficiency
            messages=[
                {"role": "system", "content": self.router_prompt.format(fields=fields_list)},
                {"role": "user", "content": user_request}
            ],
            response_format=DomainClassification
        )
        return response.choices[0].message.parsed

    def _make_expert_decision(self, user_request: str, domain: str) -> UniversalDecision:
        """Step 2: Apply domain-specific logic to make the decision."""
        guidelines = DOMAIN_REGISTRY.get(domain, DOMAIN_REGISTRY["General/Other"])
        
        response = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": self.expert_prompt.format(domain=domain, guidelines=guidelines)},
                {"role": "user", "content": user_request}
            ],
            response_format=UniversalDecision
        )
        return response.choices[0].message.parsed

    def process_request(self, user_request: str) -> dict:
        """End-to-end processing of a user request."""
        print(f"\n🔍 Analyzing Request: '{user_request}'")
        
        # Step 1: Classify
        classification = self._classify_domain(user_request)
        print(f"📂 Routed to Domain: {classification.field_of_study} (Confidence: {classification.confidence:.2f})")
        print(f"💡 Routing Reason: {classification.reasoning}")
        
        # Step 2: Decide
        decision = self._make_expert_decision(user_request, classification.field_of_study)
        
        # Step 3: Format Output
        return {
            "classification": classification.model_dump(),
            "decision": decision.model_dump()
        }


# 4. TEST ACROSS MULTIPLE FIELDS OF STUDY
  if __name__ == "__main__":
    engine = UniversalDecisionMaker()

    test_requests = [
        # 1. Medicine
        "A 45-year-old male presents with sudden, severe chest pain radiating to the left arm and shortness of breath. What is the immediate decision?",
        
        # 2. Finance
        "I have $50,000 in savings, a moderate risk tolerance, and want to retire in 15 years. Should I put it all in a single tech stock?",
        
        # 3. Software Engineering
        "We are building a real-time chat application expecting 1 million concurrent users. Should we use WebSockets with a monolithic Node.js backend or a microservices architecture with Redis pub/sub?",
        
        # 4. Law
        "My neighbor built a fence 2 feet over my property line. I want to tear it down myself this weekend. What should I do?",
        
        # 5. Education
        "I am a 10-year-old who wants to understand how black holes work. Explain it and give me a decision on what to learn next."
    ]

    for i, req in enumerate(test_requests, 1):
        print(f"\n{'='*60}")
        print(f"TEST CASE {i}")
        print(f"{'='*60}")
        
        result = engine.process_request(req)
        dec = result["decision"]
        
        print(f"\n✅ DECISION TYPE: {dec['decision_type']}")
        print(f"🎯 ACTION REQUIRED: {dec['action_required']}")
        print(f"⚠️ RISK LEVEL: {dec['risk_level']}")
        print(f"👤 HUMAN REVIEW NEEDED: {'YES 🚨' if dec['human_in_the_loop_required'] else 'NO ✅'}")
        print(f"🧠 REASONING:\n{dec['reasoning']}")
        print(f"📚 PRINCIPLES: {', '.join(dec['citations_or_principles'])}")