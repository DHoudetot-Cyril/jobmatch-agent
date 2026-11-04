"""Assembly of the hierarchical graph (public interface documented in `jobmatch.state`).

                         ┌──> analysis_team  (parse_offer ‖ parse_cv) ──┐
    START ─> supervisor ─┼──> matching_team  (score_match → analyze_gaps) ┼─> supervisor
                         ├──> writing_team   (write_letter ⇄ review_letter)
                         ├──> human_review   (interrupt) ───────────────┘
                         └──> END

Every team is a compiled subgraph used as a node; each one hands control back to the
supervisor, which routes deterministically (see `jobmatch.graph.supervisor`).
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from jobmatch.graph.human_review import human_review
from jobmatch.graph.supervisor import supervisor
from jobmatch.graph.teams.analysis import build_analysis_team
from jobmatch.graph.teams.matching import build_matching_team
from jobmatch.graph.teams.writing import build_writing_team
from jobmatch.state import JobMatchState

TEAMS = ("analysis_team", "matching_team", "writing_team")


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    g = StateGraph(JobMatchState)
    g.add_node("supervisor", supervisor)
    g.add_node("analysis_team", build_analysis_team())
    g.add_node("matching_team", build_matching_team())
    g.add_node("writing_team", build_writing_team())
    g.add_node("human_review", human_review)
    g.add_edge(START, "supervisor")
    for node in (*TEAMS, "human_review"):
        g.add_edge(node, "supervisor")
    return g.compile(checkpointer=checkpointer, name="jobmatch")


def checkpoint_serializer() -> SerializerProtocol:
    """Checkpoint serializer that explicitly allows our Pydantic/Enum state types.

    LangGraph warns (and will soon refuse) when deserializing unregistered types
    from a checkpoint; registering them keeps resume-after-interrupt working.
    """
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    from jobmatch import schemas

    allowed = [
        ("jobmatch.schemas", name)
        for name in dir(schemas)
        if isinstance(getattr(schemas, name), type)
        and getattr(schemas, name).__module__ == "jobmatch.schemas"
    ]
    return JsonPlusSerializer(allowed_msgpack_modules=allowed)
