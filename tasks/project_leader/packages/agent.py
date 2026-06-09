"""Server entry point — delegates to leader_agent.main()."""

from tasks.project_leader.packages.leader_agent import (  # noqa: F401
    CONFIG_FILE,
    CFG,
    DEBUG_FRAME,
    STATUS,
    main,
)
