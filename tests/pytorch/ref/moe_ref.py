class MoERouterConfig:
    def __init__(self, seqlen: int, experts: int, groups: int, selected_groups: int, topk: int):
        self.seqlen = seqlen
        self.experts = experts
        self.groups = groups
        self.selected_groups = selected_groups
        self.topk = topk
