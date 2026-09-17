export const meta = {
  name: 'backup-inventory-for-a0',
  description: '并行清点五棵目录树，决定哪些要备份到 A0 机器',
  phases: [
    { title: 'Inventory', detail: '五个方向各自清点并给出取舍建议' },
    { title: 'Critic', detail: '完整性批评：还漏了什么' },
  ],
}

const SCHEMA = {
  type: 'object',
  properties: {
    include: {
      type: 'array',
      description: '建议纳入备份的条目',
      items: {
        type: 'object',
        properties: {
          path: { type: 'string', description: '绝对路径（可以是目录或 glob）' },
          approx_size: { type: 'string', description: '粗略大小，如 120K / 3.4M' },
          why: { type: 'string', description: '为什么明天在 A0 上会用到它（一两句）' },
          priority: { type: 'string', description: 'must / nice / optional' },
        },
        required: ['path', 'why', 'priority'],
      },
    },
    exclude: {
      type: 'array',
      description: '明确建议排除的大块或危险内容',
      items: {
        type: 'object',
        properties: {
          path: { type: 'string' },
          approx_size: { type: 'string' },
          why: { type: 'string', description: '为什么不要带（太大 / 可重新生成 / 含凭据 / 对面已有）' },
        },
        required: ['path', 'why'],
      },
    },
    secrets_found: {
      type: 'array',
      description: '扫到的任何凭据/token/密钥所在路径，必须单独列出',
      items: { type: 'string' },
    },
    notes: { type: 'string', description: '其它需要主 agent 知道的事' },
  },
  required: ['include', 'exclude', 'secrets_found', 'notes'],
}

const CONTEXT = `
背景：用户今天要回到 A0 机器（heliosr-1b114-c07-1，单张 gfx1250、VR 限频 1100MHz）继续做
Primus-Turbo 的 attention 内核优化。过去两天的工作在本机（B0 / ctheliosp-1b112-a37-1，4 卡满频）上做。
现在要把「在 A0 上继续干活会用到的东西」打成一个尽量小的 tar.gz 传过去。

判断标准：
- MUST：没有它明天会重做一遍工作，或者丢了就找不回来（唯一副本）。
- NICE：能省时间但可重新生成。
- 排除：体积大且可重新生成（编译缓存、构建产物、venv、.git objects 重复、ISA 原始 dump 中的重复部分）、
  A0 上本来就有的（git 仓库本身、公共镜像）、以及任何凭据。
- 注意 A0 上已经有：同一个 Primus-Turbo git 仓库（但落后本机 56 个提交）、同样的容器镜像、
  昨天 9-13 的工作产物（output/0913__opt_plan__claude 是在 A0 上产生后同步过来的，A0 可能已有旧版本）。

只做清点和判断，**不要拷贝、不要创建任何文件、不要修改任何东西**。
用 du/ls/find/head 等只读命令。给出的 path 用绝对路径。
体积务必核实（du -sh），不要猜。
`

phase('Inventory')

const DIRECTIONS = [
  {
    key: 'repo',
    prompt: `${CONTEXT}

你的方向：**Primus-Turbo 仓库本身**，/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo

要回答：
1. 代码改动怎么带最省体积又最完整？分支是 gfx1250-attn-dispatch-and-tuning，领先 origin/main 56 个提交。
   对比三种方案的体积：(a) \`git bundle create x.bundle origin/main..HEAD\`，
   (b) \`git format-patch origin/main..HEAD\` 系列 patch，(c) 整个工作区 tar。
   实际跑 bundle 到 /tmp 量一下大小（/tmp 里可以写，那不算备份目录）。注意 3rdparty/hipkittens 是
   未初始化或损坏的 submodule，git 命令会报错，想办法绕过。
2. 工作区里有没有**未提交**的改动或未跟踪文件是明天需要的？（git status，注意 submodule 报错）
   根目录有个 output.tar.gz，看看那是什么。
3. output/ 下四个目录（0913__opt_plan__claude 1.3M、0914__campaign 7.1M、
   0914__hk_udna1_gate 76K、0914__repro__c07 288K）逐个看内容，哪些是 must。
   特别注意 0914__campaign 的 7.1M 里大头是什么（ledgers/ logs/ t7/ bin/），有没有可以砍的。
4. tools/gfx1250/ 下的脚本是否都已提交进 git（已提交的就不用单独带）。

返回结构化结果。`,
  },
  {
    key: 'opevolve',
    prompt: `${CONTEXT}

你的方向：**op-evolve 的运行产物**，/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts
总共 668M，其中 rounds/ 占 635M、job_context/ 占 34M。这肯定不能整个带走。

要回答：
1. 逐层 du 下去，搞清楚 635M 到底是什么占的（每轮的 op/ 拷贝？scratch/？raw/ ISA dump？built 产物？）。
2. 哪些是**不可再生的知识**：progress.md、note.md、每轮的 opt.md / reflect.md、
   findings/（facts.md、dead_ends.md、pool.md、route.md）、timing.yaml、gate.log、state.yaml、
   以及那些静态 census 的结果（.amdgcn 的统计，不是 .amdgcn 本身）。把这些的总体积量出来。
3. 如果只带这些「文本知识」，op-evolve 还能不能 resume？如果不能，说明白代价——
   明天在 A0 上是打算重新起一个 job 还是续跑？给出两种方案各自需要带什么、多大。
4. op-evolve 仓库本身（不含 artifacts）多大？A0 上有没有？（/home/lihuzhan/code/2026_0910__op-evolve 是不是 git 仓库、有没有 remote）
5. 还有 job spec yaml（在 Primus-Turbo 的 output/0913__opt_plan__claude/phase0/ 下）和本机改过的版本，
   以及 ~/.venv-op-evolve —— venv 不要带，但要记下它是怎么建的。

返回结构化结果。`,
  },
  {
    key: 'claude',
    prompt: `${CONTEXT}

你的方向：**Claude 相关文件**，/home/lihuzhan/.claude/

已知体积：projects 29M、plugins 7.4M、file-history 988K、cache 668K、backups 224K、
session-env 36K、plans 24K、skills 20K、history.jsonl 12K、sessions 12K、settings.json 4K。

要回答：
1. projects/-home-lihuzhan-code-2026-0903--turbo-Primus-Turbo/ 下面：
   44582308-....jsonl 是 7.9M 的会话记录（今天这整场对话），同名目录也是 7.9M——
   看看目录里是什么，两者是否重复。memory/ 里有什么。
   这个 jsonl 值不值得带？（它是完整的推理与实验过程，但 7.9M 且压缩率可能很高——实测 gzip 后多大）
   projects/ 下还有没有别的项目目录，跟这次工作有关吗？
2. skills/gpu-kernel-campaign/ —— 这是昨天整理的经验 skill，必带。确认内容和体积。
3. plans/ 下的计划文件、settings.json、session-env/ —— 哪些带过去有用？
   **特别注意 settings.json 和 session-env 里有没有 API key / token**。
4. plugins/ 7.4M 是什么，需要带吗？
5. 全盘扫一下 ~/.claude 下有没有凭据文件（grep -rIl 一些关键词，但**不要把 key 的内容打印出来**，只报路径）。
6. 另外看看 ~/claude/.amd_llm_env 和 ~/.codex/config.toml 是否存在、是不是含凭据。
   这两个用户提过，但含凭据的东西**不应该进备份包**——确认后在 secrets_found 里列出路径即可。

返回结构化结果。`,
  },
  {
    key: 'siblings',
    prompt: `${CONTEXT}

你的方向：**其它几个 code 目录**，判断它们是不是本机独有、需不需要带。

/home/lihuzhan/code/ 下有：2026_0828__primus/、2026_0903__kyle_flydsl/、2026_0903__turbo/、
2026_0910__op-evolve/、2026_0911__kyle_skill/、aiter-src/

要回答（每个目录）：
1. 多大？是不是 git 仓库？有没有 remote（有 remote 就说明 A0 上可以自己 clone，不用带）？
2. 有没有**未提交的本地改动**（这才是唯一副本、必须带的东西）？
   用 \`git -C <dir> status --porcelain\` 和 \`git -C <dir> log --oneline origin/HEAD..HEAD\` 之类判断。
3. 2026_0903__turbo/ 下除了 Primus-Turbo 还有别的吗？
4. aiter-src 是 aiter 的源码，本次用到了它的 gfx1250 预编译 .co 和 ABI 解析——
   里面有没有本机产生的分析产物（反汇编、ABI 提取结果）是唯一副本？
5. 3rdparty/hipkittens 这个 submodule 在 Primus-Turbo 里的状态如何？
   （昨天说本机未初始化、是从 A0 rsync 过来的，所以 A0 上本来就有——确认一下）

返回结构化结果。`,
  },
  {
    key: 'rig',
    prompt: `${CONTEXT}

你的方向：**运行环境的可复现性** —— 明天在 A0 上要把这套测量装置重新搭起来，需要哪些信息/文件。

要回答：
1. 本机跑过哪些容器？（\`docker ps -a\`、\`docker images\`）用的是哪个镜像、什么 tag、
   启动参数是什么（\`docker inspect\` 里的 Cmd/Entrypoint/Mounts/Env，**注意不要把 Env 里的密钥打出来**）。
   把「怎么起这个容器」总结成可直接复制的命令。A0 上有没有同一个镜像不确定，所以镜像名和 tag 要写清楚。
2. 本机有没有为了跑通而打的本地补丁/shim：比如 rocminfo 的 PATH shim、
   pip uninstall primus_turbo 的步骤、TRITON_CACHE_DIR 的隔离约定。找找它们在哪。
3. python 虚拟环境 ~/.venv-op-evolve：多大、装了什么（\`pip freeze\` 存一份文本就够，venv 本身别带）。
4. Triton 缓存目录（/tmp/triton_cache_* 之类）有多大、要不要带（提示：编译产物，通常不带，
   但如果里面有 .amdgcn 静态 census 的原始依据、且体积可控，可以考虑）。
5. 本机有没有跑过的一次性脚本散落在 /tmp 或家目录下、是明天要用的？
   （比如 output/0914__campaign/bin/ 之外的）

返回结构化结果。`,
  },
]

const results = await parallel(
  DIRECTIONS.map((d) => () =>
    agent(d.prompt, { label: `清点:${d.key}`, phase: 'Inventory', schema: SCHEMA })
      .then((r) => (r ? { key: d.key, ...r } : null))
  )
)

const good = results.filter(Boolean)
log(`清点完成：${good.length}/${DIRECTIONS.length} 个方向返回`)

phase('Critic')

const digest = good
  .map(
    (r) =>
      `### ${r.key}\nINCLUDE:\n${r.include
        .map((i) => `- [${i.priority}] ${i.path} (${i.approx_size || '?'}) — ${i.why}`)
        .join('\n')}\nEXCLUDE:\n${r.exclude
        .map((e) => `- ${e.path} (${e.approx_size || '?'}) — ${e.why}`)
        .join('\n')}\nSECRETS: ${r.secrets_found.join(', ') || '无'}\nNOTES: ${r.notes}`
  )
  .join('\n\n')

const critic = await agent(
  `${CONTEXT}

下面是五个方向各自的清点结果。你的任务是**完整性批评**：找出还漏了什么。

${digest}

具体检查：
1. 有没有哪类东西五个方向都没覆盖到？（想想：明天在 A0 上一坐下来要做的第一件事是什么，
   需要什么才能开始——计划、验收线、上次停在哪、怎么起容器、怎么跑 harness、
   怎么复现冠军配置、以及**跨机换算系数**。）
2. 有没有「只有本机才有、丢了就没了」的东西被当成可再生而排除了？重点怀疑：
   实测数据 ledger（jsonl）、一次性诊断脚本、失败实验的记录。
3. A0 是限频 1100MHz 单卡，本机是满频 4 卡——有没有什么**只在满频机器上才测得出来**、
   必须以数据形式带过去的结论？（比如满频下的绝对毫秒数、满频天花板、验收门槛值）
4. 有没有建议 include 的东西其实是危险的（凭据）或者纯属浪费体积？
5. 提出一份「README 该写什么」的要点清单——备份包里应该有一个入口文件，
   让明天的人（或明天的 agent）一眼知道先读什么、怎么开始。

自己用只读命令去核实你的怀疑，不要只做纸上推理。**不要拷贝或创建任何文件。**`,
  { label: '完整性批评', phase: 'Critic', schema: {
    type: 'object',
    properties: {
      missing: {
        type: 'array',
        description: '被漏掉、应当补进备份的条目',
        items: {
          type: 'object',
          properties: {
            path: { type: 'string' },
            approx_size: { type: 'string' },
            why: { type: 'string' },
            priority: { type: 'string' },
          },
          required: ['path', 'why', 'priority'],
        },
      },
      wrongly_included: {
        type: 'array',
        description: '不该带的（凭据或纯浪费体积）',
        items: {
          type: 'object',
          properties: { path: { type: 'string' }, why: { type: 'string' } },
          required: ['path', 'why'],
        },
      },
      readme_outline: {
        type: 'array',
        description: 'README 应当包含的要点，按顺序',
        items: { type: 'string' },
      },
      notes: { type: 'string' },
    },
    required: ['missing', 'wrongly_included', 'readme_outline', 'notes'],
  } }
)

return { inventory: good, critic }
