"""LangGraph router for SensorDoc-AI.

This module adds a small agent layer on top of the deterministic RAG pipeline.
It routes datasheet questions into the existing retrieval pipeline and rejects
dangerous or out-of-scope requests before tool use.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from rag_pipeline import PromptInjectionGuard, RAGPipeline


class RouterState(TypedDict, total=False):
    question: str
    route: Literal["blocked", "datasheet", "out_of_scope"]
    answer: str


class LangGraphRouterAgent:
    DATASHEET_TERMS = {
        "i2c",
        "spi",
        "uart",
        "uwb",
        "ble",
        "imu",
        "mpu",
        "mpu-6000",
        "mpu-6050",
        "sensor",
        "datasheet",
        "pin",
        "register",
        "address",
        "scl",
        "sda",
        "ad0",
        "airtag",
        "accelerometer",
        "gyroscope",
        "clock",
        "protocol",
        "interface",
    }

    def __init__(self) -> None:
        self.pipeline = RAGPipeline()
        self.pipeline.load_existing()
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(RouterState)
        graph.add_node("guard_input", self._guard_input)
        graph.add_node("classify_route", self._classify_route)
        graph.add_node("answer_datasheet", self._answer_datasheet)
        graph.add_node("answer_out_of_scope", self._answer_out_of_scope)

        graph.add_edge(START, "guard_input")
        graph.add_conditional_edges(
            "guard_input",
            self._route_after_guard,
            {
                "blocked": END,
                "classify_route": "classify_route",
            },
        )
        graph.add_conditional_edges(
            "classify_route",
            self._route_after_classification,
            {
                "datasheet": "answer_datasheet",
                "out_of_scope": "answer_out_of_scope",
            },
        )
        graph.add_edge("answer_datasheet", END)
        graph.add_edge("answer_out_of_scope", END)
        return graph.compile()

    @staticmethod
    def _normalize_tokens(text: str) -> list[str]:
        return RAGPipeline._tokenize(text)

    def _guard_input(self, state: RouterState) -> RouterState:
        question = state["question"].strip()
        blocked = PromptInjectionGuard.validate_question(question)
        if blocked:
            return {"question": question, "route": "blocked", "answer": blocked}
        if not question:
            return {
                "question": question,
                "route": "out_of_scope",
                "answer": "Please ask a question about the datasheet content.",
            }
        return {"question": question}

    def _classify_route(self, state: RouterState) -> RouterState:
        question = state["question"]
        tokens = set(self._normalize_tokens(question))
        matched_terms = tokens & self.DATASHEET_TERMS

        if matched_terms:
            return {"question": question, "route": "datasheet"}

        return {"question": question, "route": "out_of_scope"}

    def _answer_datasheet(self, state: RouterState) -> RouterState:
        question = state["question"]
        answer = self.pipeline.query_sensor_info(question)
        return {"question": question, "route": "datasheet", "answer": answer}

    @staticmethod
    def _answer_out_of_scope(state: RouterState) -> RouterState:
        return {
            "question": state["question"],
            "route": "out_of_scope",
            "answer": "This agent is limited to hardware datasheet questions. Ask about the sensor, pins, interfaces, registers, or specifications.",
        }

    @staticmethod
    def _route_after_guard(state: RouterState) -> Literal["blocked", "classify_route"]:
        if state.get("route") == "blocked" or state.get("answer"):
            return "blocked"
        return "classify_route"

    @staticmethod
    def _route_after_classification(state: RouterState) -> Literal["datasheet", "out_of_scope"]:
        return state.get("route", "out_of_scope")

    def invoke(self, question: str) -> str:
        result = self.graph.invoke({"question": question})
        return result["answer"]


_agent: LangGraphRouterAgent | None = None


def agent_query(question: str) -> str:
    global _agent
    if _agent is None:
        _agent = LangGraphRouterAgent()
    return _agent.invoke(question)


if __name__ == "__main__":
    agent = LangGraphRouterAgent()
    while True:
        question = input("\nAgent question (or 'quit'): ").strip()
        if question.lower() in {"quit", "exit", "q"}:
            break
        print(f"\n{agent.invoke(question)}")