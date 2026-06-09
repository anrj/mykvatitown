"""Server entry point — re-exports leader_agent for packaging."""

from tasks.project_leader.packages import leader_agent

CONFIG_FILE = leader_agent.CONFIG_FILE
CFG = leader_agent.CFG
DEBUG_FRAME = leader_agent.DEBUG_FRAME
STATUS = leader_agent.STATUS
main = leader_agent.main
