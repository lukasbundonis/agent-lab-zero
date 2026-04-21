# two_agents.py
# This is a basic two-agent environment. Each agent has its own properties.

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Agent:
    name: str
    role: str
    context: str
    goal: str
    constraints: list[str] = field(default_factory=list)
    voice: str = ""
    relationship: str = ""

    def system_prompt(self) -> str:
        """Assemble the agent's character sheet into an XML-tagged system prompt."""
        constraints_block = "\n".join(f"- {c}" for c in self.constraints)
        
        return f"""<identity>
You are {self.name}, {self.role}
</identity>

<context>
{self.context}
</context>

<goal>
{self.goal}
</goal>

<constraints>
{constraints_block}
</constraints>

<voice>
{self.voice}
</voice>

<relationship>
{self.relationship}
</relationship>

<turn_behavior>
Respond as {self.name}, in first person. Do not narrate actions in third person.
Do not speak for the other participant. Keep each turn focused and purposeful.
</turn_behavior>"""


# ---- Agent definitions ----

AGENT_A = Agent(
    name="Vera",
    role="You are a chef who has been given a pantry of ingredients and must design a dinner menu in collaboration with a nutritionist.",
    context="You know exactly what ingredients are available but you do not know the guest's dietary restrictions. Your collaborator does. You must exchange information through conversation to design a menu that works for both constraints.",
    goal="Produce a three-course menu that uses only available ingredients and respects all dietary restrictions. You succeed when both of you agree the menu is complete.",
    constraints=[
        "Do not invent ingredients you haven't been told are available.",
        "Ask specific questions rather than making broad guesses.",
        "When you propose a dish, state which ingredients it uses.",
    ],
    voice="Warm but practical. You think out loud briefly, then commit to concrete proposals.",
    relationship="Your collaborator is Dr. Okafor, a nutritionist. You respect their expertise on restrictions but you lead on culinary decisions.",
)

AGENT_B = Agent(
    name="Dr. Okafor",
    role="You are a nutritionist working with a chef to design a dinner menu for a specific guest.",
    context="You know the guest's dietary restrictions in detail but you do not know what ingredients are in the pantry. The chef does. You must exchange information through conversation to arrive at a workable menu. The restrictions: no shellfish, no dairy, no added sugar, and the guest dislikes bitter greens.",
    goal="Ensure every course respects the guest's restrictions. You succeed when both of you agree the menu is complete.",
    constraints=[
        "Do not approve a dish until you've confirmed its ingredients against the restrictions.",
        "Raise concerns immediately rather than deferring.",
        "If a proposed dish conflicts with a restriction, suggest a specific modification.",
    ],
    voice="Precise and collegial. You ask clarifying questions before approving anything.",
    relationship="Your collaborator is Vera, a chef. You trust their culinary judgment but you are the final word on whether a dish is safe for the guest.",
)

import anthropic

client = anthropic.Anthropic()


def agent_turn(agent: Agent, history: list[dict]) -> str:
    """Run one turn for the given agent, returning just the text response."""
    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=agent.system_prompt(),
        messages=history,
    )
    return response.content[0].text


def run_conversation(
    agent_a: Agent,
    agent_b: Agent,
    opening_message: str,
    max_turns: int = 6,
) -> list[tuple[str, str]]:
    """
    Run a back-and-forth conversation between two agents.
    Returns a transcript as [(speaker_name, text), ...].
    """
    # Each agent has its own history. The SAME utterance appears in both,
    # but with different role labels depending on whose view we're in.
    history_a: list[dict] = []
    history_b: list[dict] = []

    # Seed Agent A with the opening. From A's POV, this is a user speaking to them.
    history_a.append({"role": "user", "content": opening_message})

    transcript: list[tuple[str, str]] = []
    transcript.append(("Moderator", opening_message))

    for turn in range(max_turns):
        # --- Agent A's turn ---
        a_reply = agent_turn(agent_a, history_a)
        # From A's POV, A just spoke (assistant). From B's POV, someone else spoke (user).
        history_a.append({"role": "assistant", "content": a_reply})
        history_b.append({"role": "user", "content": a_reply})
        transcript.append((agent_a.name, a_reply))
        print(f"\n[{agent_a.name}]\n{a_reply}")

        # --- Agent B's turn ---
        b_reply = agent_turn(agent_b, history_b)
        # Mirror image: B's POV is assistant, A's POV is user.
        history_b.append({"role": "assistant", "content": b_reply})
        history_a.append({"role": "user", "content": b_reply})
        transcript.append((agent_b.name, b_reply))
        print(f"\n[{agent_b.name}]\n{b_reply}")

    return transcript


if __name__ == "__main__":
    opening = (
        "You two have been asked to design a three-course dinner menu together. "
        "Vera, you know what's in the pantry. Dr. Okafor, you know the guest's "
        "restrictions. Begin."
    )

    transcript = run_conversation(
        agent_a=AGENT_A,
        agent_b=AGENT_B,
        opening_message=opening,
        max_turns=6,
    )

    print("\n\n--- FULL TRANSCRIPT ---")
    for speaker, text in transcript:
        print(f"\n[{speaker}]\n{text}")