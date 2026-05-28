# 学术论文内容提取
来源文件: your_academic_paper.pdf


--- 第 1 页 ---

Towards Automating Data Access Permissions in AI Agents
Yuhao Wu∗, Ke Yang†, Franziska Roesner‡, Tadayoshi Kohno§, Ning Zhang∗, Umar Iqbal∗
∗Washington University in St. Louis †University of California, Irvine ‡University of Washington §Georgetown University
Abstract—AsAIagentsattempttoautonomouslyactonusers’ Fundamentally, the key issues are limited user trans-
behalf, they raise transparency and control issues. We argue parencyandcontroloverthedataaccessbyAIagents.Prior
thatpermission-basedaccesscontrolisindispensableinprovid- computingsystems,suchasmobileplatforms,haveencoun-
ing meaningful control to the users, but conventional permis- teredsimilarproblemsandhavereliedonpermissionmodels
sionmodelsareinadequatefortheautomatedagenticexecution (inwhichusersareaskedtograntapplicationspermissionto
paradigm.Wethereforeproposeautomatedpermissionmanage- accesssensitiveresources)toprovideuserstransparencyand
mentforAIagents.Ourkeyideaistoconductauserstudyto control over data usage in the system [21], [22], [26], [67],
identifythefactorsinfluencingusers’permissiondecisionsand [68].Similarly,AIagentswillneedpermissionmanagement
models to control access to resources.
to encodethese factors intoan ML-based permission manage-
Permission management, however, needs to be designed
mentassistantcapableofpredictingusers’futuredecisions.We
anew for AI agents, where new challenges arise that stretch
find that participants’ permission decisions are influenced by
the limits of existing permission models. For example, as
communicationcontextbutimportantlyindividualpreferences
AI agents generate new functionalities based on input from
tendtoremainconsistentwithincontexts,andalignwiththose
varioussystemmodules,dataneededtoresolveuserqueries
of other participants. Leveraging these insights, we develop a
may not be known beforehand, thus diminishing the utility
permissionpredictionmodelachieving85.1%accuracyoverall
of legacy install time permissions. Similarly, as AI agents
and 94.4% for high-confidence predictions. We find that even
may need several pieces of user information to resolve a
withoutusingpermissionhistory,ourmodelachievesanaccu-
user query, constantly interrupting users for permissions is
racy of 66.9%, and a slight increase of training samples (i.e.,
incompatible with the automation promised by AI agents,
1–4) can substantially increase the accuracy by 10.8%.
thusdiminishingtheutilityofstandardruntimepermissions.
Considering that automation is a key value proposition
1. Introduction of AI agents, a permission management system that can
automaticallymakedecisionsonusers’behalfiscriticalfor
AI agents. Building an automated permission management
LLMs have enabled a new computing paradigm, in requires addressing two important problems. First, under-
which the system (referred to as an AI agent or an agentic standingthefactorsthatusersconsiderorfactorsthatother-
system) relies on machine learning models to autonomously wise influence users’ permission decision-making. Second,
resolve user queries expressed in natural language [76]. making permission decisions that precisely meet user needs
For example, to resolve a user query to book a flight, the and expectations for future data sharing, including previ-
AI agent might engage with the necessary tools (e.g., a ously unseen data types. This paper takes a multifaceted
travel reservation tool), automatically access the required approach, including both (i) conducting a user study to
user information (e.g., from the system storage/memory), understand user permission preferences and (ii) exploring
use the user’s credit card details, and make the booking. the design of a permission prediction system along with its
While this execution paradigm is tremendously powerful implementation and evaluation.
and is enabling exciting use cases [7], [25], [61], there are To understand users’ data sharing permission prefer-
serious security and privacy risks to consider. ences, we conduct a vignette-based user study. We create
At a high level, a key concern is that the AI agent’s a bespoke user study setup by developing our own website
actions may not align with the user’s intentions or expecta- that attempts to immerse participants in training a futuristic
tions, which could be due to untrustworthy system modules personal assistant, including training it to share data, as per
influencingtheLLM[33],[78],[85],LLMmakingmistakes their needs. We capture differing user preferences across a
on its own [31], [44], [82], or the user simply not agreeing wide range of questions spanning several domains, such as
with the agent’s course of action [41], [54], [75]. These health and fitness, finance, and entertainment. Our findings
concerns apply to a broad range of the agent’s activities, indicate that fewer participants express permissions to AI
such as autonomous data access and actions in the real- agentsforautomaticenforcementwhentheymakemistakes,
world. In this paper, we focus on the security and privacy and participants often struggle with providing appropriate
issues pertaining to the access of data. For example, we permissions, i.e., under- and over-permissioning are com-
attempt to limit the unexpected sharing of a user’s credit monissuesinagenticsystems,similartopriorsystems[43],
card details with the wrong tool. [63], [70]. We also note that participants’ permission deci-
1

--- 第 2 页 ---

sions are influenced by the context of the communication, memory (to keep a record of user interactions and data),
choiceofdatabeingshared,andtheirprivacyconsciousness. and a set of tools (to take action in the real world) [76]. To
Importantly,wefindthat,atleastinthecontextofourstudy, resolveauserquery,AIagentsidentifytherelevanttoolsand
participants’ permission preferences remain consistent, are data, formulate an execution plan (i.e., a set of instructions)
similar across various communication contexts, and are of- for an LLM, and autonomously act on the formulated plan.
ten similar to other participants—presenting an opportunity For example, for a user query to “book a flight”, the
to predict their future permissions. agent will first determine that it needs to use a travel
Tolearnandpredictusers’datasharingpermissionpref- reservationtoolandrequirestheuser’sinformation(suchas
erences, we explore developing a hybrid machine learning the user’s name and date of birth). The execution plan may
framework that learn from individual user preferences and include instructions to search for flights, provide payment
also from preferences of similar users (using data from our details, and make the booking. Acting on the plan will
user study). We leverage LLM in-context learning [57] to include the agent calling the appropriate travel reservation
learn individual user preferences for two crucial reasons: toolAPIswiththerelevantdata,listedintheexecutionplan.
(i) we possess limited permission decision history for each AIagentsexhibitvaryingdegreesofautonomy;insome
user, and LLMs have demonstrated attaining high accuracy cases, they can complete tasks entirely of their own, while
with limited training data, i.e., “few shots” [19], [57]; (ii) in other cases they require some level of human supervi-
permission decisions need to be made for unseen data, sion during execution. For example, for the flight booking
which LLMs allow to make without retraining [13]. We request, agents may retrieve trip dates and locations from
leverage collaborative filtering [29], [30] because it allows the user query, the user’s name and date of birth from the
us to learn from the preferences of other similar users. Our memory storage, and may only request the user to provide
final resulting model combines both in-content learning and credit card information.
collaborative filtering, as they complement each other. We
achieve an accuracy of 85.1% (with a recall of 85.2% and
2.2. Security and Privacy Risks
a precision of 92.8%). We also explore adjusting prediction
confidencescorethresholdsandfindthatastricterthreshold,
As the automated execution paradigm of AI agents
we can achieve an accuracy of 94.4%, but compromise
makes user interactions seamless, there are obvious benefits
on making predictions for 74.1% of the data. We also
to it; however, it also presents serious security, privacy, and
observe that even a slight increase in training data (i.e.,
safety issues to the users. At a high level, a key concern
user permission decision history), our classification accu-
is that the AI agent’s actions may not align with the user’s
racy substantially increases. For example, accuracy reaches
intentionsorexpectations.Themisalignmentcouldbedueto
66.9% without permission history, but incorporating history
a variety of reasons, such as untrustworthy system modules
fromjust1–4queriesimprovesitby10.8%.Tofosterfuture
research, we release our user study data and code1. influencing the LLM, LLM making mistakes of its own,
or the user simply not agreeing with the agent’s course of
Our key contributions are as follows:
action. These concerns apply to a broad range of agents’
1) We propose automating data access permissions in AI
activities, such as agents using user data of their own or
agents, such that a permission assistant can observe a
taking actions on users’ behalf in the real world, which are
user’s permission decision history and can make auto-
distinct and require tailored treatment. For the scope of this
matic decisions on the user’s behalf in the future.
paper, we only focus on limiting the security and privacy
2) To realize our goal of automating permission decisions,
issues that pertain to the usage of data.
we develop a bespoke vignette-based user study to un-
Next, we describe some of the fundamental issues that
derstand various factors that may influence users’ data-
could lead to a misalignment between users’ expectations
sharing permission decisions in AI agents. We then
and AI agents’ data usage practices.
conduct the study with 205 participants on Prolific.
3) We translate the insights from our user study into a Inherent Limitations in Agentic Execution Paradigm.
permission inference framework capable of predicting AI agents enhance their capabilities through exposure to
users’permissionpreferences,achieving85.1%accuracy system resources, such as data and tools. It means that
overall and 94.4% for high-confidence predictions. only when an agent is aware of the system capabilities
(i.e., a particular tool or a piece of data) can it use those.
2. Background and Motivation Thus,agentsareoftendesignedtogetunrestrainedaccessto
systemresources,includingtheonesthatmaynotbeneeded
to address the user query, which violates the principle of
2.1. AI Agents
leastprivilege.AsLLMsaresusceptibletopromptinjection,
malicious resources (such as malicious third-party tools or
While there is no standardized AI agent architecture,
files) can exploit the LLM’s unrestrained system access to
at their core, AI agents (often also referred to as agentic
read sensitive user data or influence the LLM’s execution.
systems)consistofanLLM(typicallyaccessedviaanAPI),
Another key issue is that the LLM’s interfacing is
a system prompt (that defines the agent’s functionality),
based on natural language, which can be imprecise and
1.https://github.com/llm-platform-security/ai-agent-permissions ambiguous [34], [45]. It means that even in non-adversarial
2

--- 第 3 页 ---

scenarios, the underlying LLM in an AI agent might collect ingsystems.Specifically,theinstall-time[5]andruntime[6]
incorrect or unnecessary user data. For example, if the permissions from conventional systems are insufficient, es-
travel reservation tool imprecisely specifies that it needs pecially considering the automated execution paradigm of
relevant user data to make a travel reservation, the LLM’s AI agents. For example, install-time permissions assume
interpretation of this data may be different than that of that the resources needed by applications are known be-
the tool and result in inadvertent collection and sharing of forehand, and thus users can make permission decisions
unnecessary user data. Hallucination issues in LLMs may at the time of installation. In contrast, in most cases of
further exacerbate these problems [34]. agent execution, behavior is determined at runtime based
Untrustworthy Third-party Tools.AIagentsrelyoninput on input from system modules, which can lead to the
from several system modules to determine and steer their emergence of new behaviors. Thus, the resources (which
execution. Some of these modules, chiefly tools, are de- can be very broad) needed to solve a query may not be
velopedbythird-partydevelopersandloadunvettedcontent knownbeforehand[76],[81].Similarly,runtimepermissions
fromtheinternet,whichmakesthemanunreliablesourceto (upon which smartphone platforms have largely converged)
determinethesystem’sexecution.Forexample,amalicious, allow users to manually make decisions on access of indi-
compromised, or buggy travel reservation tool may direct vidual resources, which suits conventional systems as only
the agent to include users’ passport numbers for booking a handful of resources are accessed at runtime (and where
flights even when it is not necessary (such as for domestic user interactions already follow a typical UI flow), such
flights). Prior research has already shown that third-party as granting location access permission in mobile platforms.
tools often collect more data than is needed, including Whereas agents often require accessing several pieces of
sensitive data prohibited by the platforms [78]. While the user data at runtime to solve a query, thus merely applying
problem of excessive data collection by third-party services legacy runtime permission models can degrade the user
has existed in prior computing platforms [20], [32], [65], experience and contribute to permission fatigue.
[74], it presents elevated risks in the case of AI agents, While conventional permission models are mostly lim-
because of their pivotal reliance on third-party tools to ited in supporting the agentic execution paradigm, they
determine their execution. could still be suitable for some use cases. For example, AI
agents could leverage install-time permissions to manage
Users’ Privacy Consciousness. Users’ expectations, de-
OAuth-based authentication [1] in AI agents.
sires, and privacy needs may also simply not align with an
AI agent’s action. For example, prior research has shown
3. Towards Automated Permissions Manage-
that users often feel uneasy about sharing intimate data
(e.g., health, finances) with AI assistants, fearing misuse ment in AI Agents
or eavesdropping [40], [49]. Prior research also shows that
users differ significantly in their willingness to share data, Considering that a significant number of data resources
which is often influenced by context trust, tech literacy, are accessed at runtime to facilitate execution of queries—
and prior experience [2], [8], [49]. In some cases, users more than users can reasonably be asked to constantly
may refuse to avail some services or compromise on the evaluate and decide upon—we argue that a permission
user experience. For example, in the context of the flight management system that can automatically make decisions
booking example, users may prefer providing the origin on users’ behalf is a necessity for agents. Our observation
flightcitymanually,insteadofhavinganagentinferitfrom is consistent with previous work on voice-based personal
the device’s GPS sensor. assistants, which found that while users want control, they
dislike excessive prompts and prefer automated permission
2.3. Permission-Based Data Access management with minimal interruptions [49], and permis-
sionpromptfatiguehasbeenalongstandingknownissuein
Fundamentally, the key issues are limited transparency other contexts as well [11], [22], [53], [72].
andcontroloverthedataaccessbyAIagents.Priorcomput-
ing systems and platforms, such as mobile platforms, have 3.1. Research Goals
encounteredsimilarproblemsandhavereliedonpermission
models to provide user transparency and control over the Developing automated permission management requires
data usage in the system [21], [22], [26], [67], [68]. Like- addressing three key challenges: (i) understanding diverse
wise,agentscanalsobenefitfromapermissionmanagement user data sharing preferences, (ii) accurately learning and
model to control access to resources. predicting user preferences, and (iii) reliably enforcing pre-
Deployed AI agents, such as ChatGPT, currently adapt dicted preferences. In this paper, we focus on (i) and (ii);
permission models from conventional systems [58]; how- prior complementary work [8], [52], [79] can be used to
ever, as we describe next, they fall short in supporting and address (iii). We describe our approaches to achieve these
instead hinder the automated agentic execution paradigm. goals below.
Insufficiency of Existing Permission Models. Permission
management systems for AI agents need to be developed 3.1.1. Goal 1: Understanding diverse user preferences.
anew as they significantly differ from conventional comput- A prerequisite to making permission decisions on users’
3

--- 第 4 页 ---

behalfistounderstandthepreferencesandexpectationsofa However, prior work can only predict a handful of known
varietyofusers.Tounderstanduserpreferences,weconduct system resources and data types, which suited the needs of
avignette-baseduserstudy,whichpresentsseveralscenarios older systems. For instance, notable studies on non-LLM
to participants to capture their preferences. Building on the agents/assistants manage data access permissions across
priorwork,weidentifyseveralfactors,suchasthecontextof only 15 data types at a coarse granularity [4], [83], [84].
the communication, privacy consciousness, and users’ prior In contrast, LLM-based AI agents routinely encounter new
experiences, that may influence user preferences [2], [8], previously unseen data types as users explore new use
[40],[49].Ouruser-studypresentsseveralscenariostousers cases [78], and thus prior approaches simply cannot scale
to capture their preferences across a wide range of factors to AI agents. For example, a classic ML model trained on a
that may influence their preferences. setofdatatypesusedbyatool/appwouldrequireretraining
While prior work exists on understanding user prefer- to make predictions for new unseen data types used by the
ences and expectations for personal assistants, it mostly tool/app. Our proposed approach tackles this challenge by
focused on older non-LLM technologies [2], [40], [47], relyingon LLMin-contextlearning,which doesnotrequire
[48], [49]. Newer LLM-based personal assistants/agents retraining to make permission decisions on unseen data.
are fundamentally different as they rely on a new natural
language-based automated execution paradigm and offer far 3.2. Threat Model
more advanced capabilities. For example, non-LLM agents
mostly supported single-dialog user interactions and carried
System Model. We assume AI agents that maintain user
out limited tasks automatically; consequently, prior studies
collected data in dedicated memory modules and attempt to
only analyzed a handful of scenarios while examining user
automatically use that data to provide personalized services
permission preferences [49]. Whereas LLM-agents support
to the users. These AI agents include well-known personal
multi-dialog user interactions and carry out many tasks
assistants, such as OpenAI’s ChatGPT [60], as well as AI
automatically [50], [81]. Users are likely to form different
agents developed through agent development toolkits, such
mental models for modern LLM-based AI agents than for
as LangChain [28] and LlamaIndex [35]. These AI agents
traditional, non-LLM agents, so tailored user studies are
also support third-party tools, and both the AI agent and
essential to capture these distinct experiences. To the best
third-partytoolscancollectanduseuserdata.AIagentsalso
of our knowledge, we are the first to study user preferences
include system scaffolding that allows them to control the
in automating data sharing permissions in the context of
execution flow, e.g., accessing data from memory, initiating
LLM-based AI agents. Our goal is also not just to sim-
LLMs, and making network requests via tools [76].
ply understand user preferences, but to translate them to
a system design and explore their utility in automatically Adversaries and Goals. We assume that the third-party
predicting permission decisions. Prior work has also not tools could be untrustworthy/dishonest, malicious, or com-
explored understanding user permission preferences in AI- promised (e.g., via a prompt injection). The attacker’s goals
based systems for the purposes of training an automatic are to leverage third-party tools to steal sensitive data that
permission prediction assistant. is present in the agent’s memory or exists with other tools.
We assume the LLM to be not malicious but error-prone
3.1.2. Goal 2: Accurately learning user preferences. To andmakingmistakesinaccessingunnecessaryandsensitive
enforce user preferences across a wide spectrum of conver- data, e.g., due to ambiguity of natural language [34], [45].
sational contexts and a range of data types, it is crucial to
Trust Relationships. We assume that the AI agent and its
develop a permission preference prediction system that ac-
scaffolding to be trustworthy, and do not have any direct
curately learn users’ preferences from a handful of contexts
intent to harm users (though they are still vulnerable to
and applies to other unseen contexts. More specifically, in
attacks, e.g., prompt injection). We also assume that the
a real world setting, users may only provide preferences
users may not share data with the AI agents and tools
for a select few scenarios, and the system may encounter
when they think it is unnecessary or simply do not feel
scenarios that it has not seen before. To tackle these chal-
comfortable sharing that data, despite data being necessary
lenges, we rely on a combination of collaborative filter-
for the AI agents and tools to provide functionality.
ing[29]andLLM-basedin-contextlearning[57]todevelop
our permission inference system. We choose collaborative Permission Assistant Design Assumptions. To understand
filtering because it enables learning a user’s preferences by user preferences, we communicate to users (in our user
analyzing the preferences of similar users [29], [30], thus study) that the tools may be malicious, compromised, or
avoidingthedatasparsityissues.WechooseLLMin-context dishonestly collect more data than they need. We envision
learning, as it can allow the systems to continuously refine the deployment of our permission assistant in the system
permissioninferenceandpredictnewpreviouslyunseendata scaffolding of the AI agent, without the LLM having direct
types,withoutrequiringre-training,asnewcontextsanddata access to the permission assistant. As the permission assis-
types are seen [19], [57]. tant makes probabilistic decisions, it may unintentionally
Prior work has proposed predicting user permissions in make incorrect predictions in some cases. However, the
the context of mobile [12], [23], [46], [51], [80], IoT [9], permission assistant does not act with malicious intent, and
[15], [71], and non-LLM personal assistants [4], [83], [84]. users retain control over whether to accept its decisions.
4

--- 第 5 页 ---

Based on these considerations, we assume the permission questions that prompts the participants to be careful, as the
assistant to be trustworthy. agent or tools may collect incorrect or unnecessary data.
Outofscope.Astheimplementationanddeploymentofthe Additionally, we develop our own custom website to make
permissionassistantrequiressolvinguniquechallengesofits the experience more immersive for users3.
own, we do not consider it in the scope of this paper. For
example,asecurepersonalassistantmayrequiresandboxed 4.1.2. Question curation. We develop an LLM-based
orTEE-baseddeployment,detectionofmaliciousdataflows, framework to curate questions for our user study. We begin
and control of the generation of LLM instructions. We by selecting a set of 8 domains4 and curating 21 tools
envision that the prior work on these topics in the context (spanning all domains) to record participant preferences
of AI-agents (e.g., [8], [16], [52], [79]) can be extended to on a variety of topics. Our tool curation involves listing
implement a secure permission assistant. functionalities offered by the tools and the data needed to
provide those functionalities, leveraging prior work on tool
4. Understanding User Preferences curation [17], [78]. We treat the data needed by tools as
ground truth.
To automatically make permission decisions on users’ We iteratively provide domains and their tools to the
behalf, it is crucial to first understand how users make LLM,andpromptittogeneratequeriesthatusersmightask
permission decisions in the context of agentic systems. To anAI agent,and that require usingone or more tools.Once
this end, we conduct a user study to understand users’ queries are generated, three members of our research team
permission decision process. Our goal with the user study review the generated queries to filter out redundant queries
is to understand factors that influence users’ decision mak- and semantically group similar data types (e.g., address and
ing, so that they can help inform the design of automated location). In summary, we curate 65 questions5 spanning 8
permission management systems. domains, involving 142 unique data types and 75 generic
data types across 21 tools.
4.1. Study Design
4.1.3. Primingparticipants.Beforeparticipantspursuethe
4.1.1. Overview. We consider several variables in our user questionsintheuserstudy,weprimetheparticipantsbyask-
study (such as demographics, privacy consciousness, usage ing them questions about their privacy consciousness (e.g.,
context)thatpriorresearchhasidentifiedtoinfluenceusers’ the importance of privacy to the participants, participants’
permission decisions in other computing platforms [2], [8], key privacy concerns). Prior research has shown that asking
[49]. Meanwhile, AI agents introduce unique capabilities about privacy attitudes at the start of a study can raise
that users have not experienced in prior systems. To help participants’ privacy awareness and prompt more cautious
users become familiar with these capabilities and their as- behavior throughout the task [18], [66], [73]. Our goal with
sociated risks, and to examine the influence of all factors in priming is to make participants privacy conscious, so that
acontrolledsetting,wedevelopavignette-basedstudy[24], their choices more accurately reflect their privacy posture,
which has been used in prior work investigating users’ as it may in real life when they are training an assistant to
privacy preferences [27], [38], [39], [42], [55], [69]. In make permission decisions on their behalf.
our study, we present vignettes to users that attempt to
immerse the participants in training a futuristic personal 4.1.4. Participant recruitment. We recruit a total of 205
assistant to their needs. Specifically, we present scenarios participants (we remove 2 participants because of invalid
to participants, where they are asked to help the personal responses) from Prolific [64] in the US, with a minimum
assistant: (i) pick the right set of tools to solve a query, Prolific prior survey approval rate of 90% and age over 18
(ii) pick the right set of data to solve a query, and (iii) years. Our study takes approximately 18 minutes to com-
automatically make data accessing and sharing permission plete,andwepay$4.20toeachparticipant(using$14asthe
decisions on their behalf. We present a total of 5 questions hourlywageattheleadinstitution).Ourstudywasreviewed
for tool and data selection, and 20 questions for permission byourinstitution’sreviewboard(IRB)anddeemedexempt.
decisions.2 Toinvestigatehowpotentialadversarialmanipu-
lations or mistakes may influence users’ decisions, for 25% 4.2. Findings
(5) of permission decision questions, we include incorrect
(unnecessary) data types and ask participants to express 4.2.1. When AI agents make mistakes, fewer partici-
their data sharing permission preferences. (Our analyses in pants express data sharing permissions, but more par-
Section 4.2.1 and 4.2.2 consider both incorrect and correct ticipants express not sharing permissions. As we explore
data, while the remaining sections only consider correct automating users’ data sharing permissions decisions, we
data.)Wealsoincludeawarningforallpermissiondecision
3.Wepresentthescreenshotsofouruserstudywebsiteathttps://github.
2.Wepresent4optionsforpermissiondecisions:(1)Yes,alwaysshare, com/llm-platform-security/ai-agent-permissions/blob/main/website.pdf.
(2)Yes,butaskmenexttime,(3)No,butaskmenexttime,and(4)No, 4.Entertainment, Health & Fitness, Smart Home, Travel, Shopping,
never share. These options are the same as the permission options used Work&Productivity,Social,andFinance.
by OpenAI’s custom GPTs [59] (except for never share, which is not an 5.Thequestionsaredetailedathttps://github.com/llm-platform-security/
optioninOpenAI’secosystem). ai-agent-permissions/blob/main/queries.json.
5

--- 第 6 页 ---

1.0 1.0
ants0.8 ants0.8 90% 18.7%
p p
ci ci
arti0.6 arti0.6
p p
on of 0.4 32% AAllwwaayyss/ sNheavreer share on of 0.4 90%
Fracti0.2 1275..26%% 21.2% NAAllewwvaaeyyr sss/ hsNhaearvereer  (sfahlasree) (false) Fracti0.2 15.4% UOnvdere-rp-peremrmisissisoinon
Never share (false) Appropriate permission
0.0 0.0
0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0
Percentage of data for a participant Percentage of data for a participant
Figure1:Distributionofparticipantpermissionpreferences. Figure2:Distributionofunder-,over-,andappropriatedata
The false label indicates data sharing instances where un- sharing permission rate of each participant.
necessary data types were presented to participants (along
with the necessary data types).
4.2.2. We observe that over-permissioning is substan-
tially more common than under-permissioning. We also
note that while participants struggle with providing
first investigate permission preferences that participants ex- appropriate permissions and often over-share, they are
pressed to the AI agent. We consider yes, always share mostly cautious about sharing sensitive data. Under-
and no, never share permissions, as permissions expressed and over-permissioning have been persistent problems in
for the AI agent to learn participant preferences. Figure 1 prior systems, such as mobile platforms, where users often
presents a distribution of participants and their permission granted too many or too few permissions due to poor un-
preferences. We note that 95.1% of the participants express derstanding, misleading interfaces, or decision fatigue [22],
always share permission decisions to AI agents for one or [37], leading to privacy risks and/or broken functional-
more data sharing decisions. The proportion of permission ity[3],[77].Motivatedbythesechallenges,wenextexamine
decisionsishigherforsharingdatathanfornotsharingdata, whether similar issues arise in the context of AI agents,
whichsuggeststhatusersmayengageinover-permissioning where the opacity of AI agents’ execution may make per-
(discussed in Section 4.2.2). Specifically, 82.8% and 68.0% mission alignment even more challenging.
of participants let AI agents to automatically share and not Figure 2 presents the distribution of under-, over-, and
share data at least once, respectively. appropriate data sharing permission rates of each user.6
We find that only 8.9% of participants never engage in
Sincereal-worldAIagentscanmakemistakesandover-
under-permissioning and 3.0% of participants never engage
collect data, we next explore how user permission prefer-
in over-permissioning. We observe that over-permissioning
ences change when AI agents make mistakes. Dotted lines
is substantially more common than under-permissioning,
inFigure1presentdistributionsofparticipantsandtheirper-
where 90% of participants engaging in over-permissioning
missionpreferenceswhentheAIagentmakesmistakes.We
for 15.4% or more data sharing requests they encounter.
find that participants tend to express more permission deci-
Noneoftheparticipantsgiveappropriatepermissionsforall
sionsfornotsharingdata(i.e.,nevershare).Specifically,the
of the data types on which they were prompted, and only
number of participants who never express never share per-
18.7% of the users give appropriate permissions for 90% or
missiondecisiondecreasesfrom32%(whenparticipantsare
more data types on which they were prompted.
presented with necessary data) to 21.2% (when participants
Through more in-depth analysis, we find that partici-
are presented with unnecessary data, along with necessary
pants’ data sharing behavior is impacted by the sensitivity
data). This suggests that AI agent mistakes prompt users to
of data. For example, highly sensitive information such as
assess their permission decisions, and many users are able
SSN exhibits an under-permission ratio of about 46.3%,
to identify unnecessary data permissions, and convey to the
meaningparticipantsarecautiouswithsharingsuchdata.In
AI agent to never share that information.
contrast, for relatively less sensitive data, such as Meeting
We note that the number of participants granting per- Details, which has an appropriate permission of 98.4%,
missions to automatically share data decreases. Specifically, participantsaremuchmorecomfortablegrantingaccess.For
the number of participants who never express always share ambiguous but non-sensitive data types, such as Workspace
permissions increases from 17.2% (when participants are Name (generally used by Slack), participants tend to over-
presented with necessary data) to 25.6% (when participants permission for 88.9% of the instances, likely because they
are presented with unnecessary data, along with necessary may be uncertain about the actual need of this information.
data). When participants do express automatic sharing/non-
sharing permissions, we do not observe substantial changes 6.Appropriate permission considers participant that provided only the
in their always share preferences. We surmise that when data that was necessary (as determined during query curation§ 4.1.2) for
solving the query, over-permission considers participants that shared data
some users observe agent mistakes, they are more cautious
thatwasnotrequiredtoresolvethequery,andunder-permissionconsiders
in expressing preferences for automatic data sharing. participantthatwithhelddatathatwasnecessarytoaddressthequery.
6

--- 第 7 页 ---

All Concerningdomain Domain Tool Always Yes(Once) No(Once) Never
MovieDatabase 58.3% 33.7% 5.9% 2.1%
Domain Always Never Always Never
Entertainment Web 34.4% 52.5% 4.9% 8.2%
AppleMusic 68.3% 30.2% 1.6% 0.0%
Entertainment 55.6% 2.9% 60.5% 2.6%
Health&Fitn. 48.9% 7.7% 44.6% 9.7% Weather 50.8% 37.9% 5.1% 6.1%
SmartHome 41.6% 9.3% 43.5% 7.6% LocalSearch 42.9% 39.0% 9.9% 8.2%
Travel TravelBooking 32.6% 45.4% 9.6% 12.5%
Travel 36.0% 10.5% 20.3% 3.8% Calendar 43.9% 43.9% 5.3% 6.9%
Shopping 32.2% 12.0% 24.0% 9.3% CloudDrive 17.8% 56.3% 9.6% 16.4%
Work&Produc. 30.8% 8.6% 27.6% 7.9% Email 34.4% 50.8% 9.0% 5.7%
Social 30.4% 8.1% 33.6% 9.9% Slack 34.9% 47.6% 7.9% 9.5%
Finance 22.2% 16.9% 22.4% 13.9% Email 27.7% 53.6% 8.3% 10.4%
Work&Produc.
Calendar 43.8% 47.4% 5.6% 3.2%
All 33.8% 10.7% 29.2% 10.8% CloudDrive 22.6% 64.5% 7.3% 5.6%
FitnessTracking 48.7% 35.4% 9.0% 6.9%
TABLE 1: Permission preferences (avg.) for always and PregnancyPal 41.0% 42.6% 6.0% 10.4%
never share permissions. Concerning domain columns con- Health&Fitn. Nutrition&Diet 46.2% 37.3% 7.5% 9.0%
HospitalBooking 54.0% 33.9% 6.5% 5.6%
taindataonlyfromparticipantswhomarkthecorresponding Calendar 54.8% 35.5% 4.8% 4.8%
domain as concerning. SMS 29.2% 51.4% 7.6% 11.7%
CloudDrive 27.9% 54.1% 13.1% 4.9%
Social
Instagram 26.2% 52.5% 14.8% 6.6%
Tinder 32.2% 50.0% 10.0% 7.7%
Additionally, we analyze some of the most frequently Banking 24.8% 50.6% 11.3% 13.2%
Finance
under-permissioned data types and find that participants TaxManagement 22.6% 50.2% 9.2% 18.0%
are less likely to share sensitive personal information. For Shopping Amazon 32.2% 45.0% 10.7% 12.0%
example, Child Name was not shared 65.1% of the time, SmartHome LightingControl 54.8% 32.3% 3.2% 9.7%
Work Information by 57.4%, and Passport Information by TABLE2:Permissionpreferencesacrossdomainsandtools.
54.0% across all data permission requests. These data types
are often essential for tasks, such as identity verification
or financial transactions. Importantly, participants who did
for only 2.9% of cases within the Entertainment domain,
not share these data types frequently reported strong pri-
but in 16.9% of cases with the Finance domain. We also
vacy concerns in relevant domains. For instance, 50.0%
observe that participants tend to give fewer always share
(15) of participants who did not share Account Password
permissions, for domains that they self-report as privacy
and 36.8% (25) who did not share SSN listed Finance
concerning. This variation is particularly noticeable for
as one of their privacy concerning domains. This suggests
TravelandShoppingdomains,wherealwayssharedecisions
that participants are making intentional privacy-conscious
drop by 15.7% and 8.2%, respectively.
choices, particularly in areas they value most. While some
of these data types, such as SSN or Passport Information, Differences Within Communication Contexts/Domains.
are often essential for providing functionality, we find that Next, we analyze the variance in participants’ permission
others, such as Child Name and Travel Details, could be preferences within domains. We present the breakdown of
reasonablysubstitutedwithdummyvaluesinsomescenarios all participants’ permission preferences for data-sharing for
to maintain functionalities while respecting users’ privacy. different tools within different domains in Table 2. Overall,
we note that participant preferences on data sharing for
4.2.3. Participants’ permission preferences vary even tools have differences within domains. Barring domains
within various contexts/domains, thus requiring finer withsingletools,EntertainmentandTravelhavethehighest
contextual considerations. Participants are also able to standard deviation of 14.2% and 10.5% for always sharing,
identify and correct AI agents’ mistakes in sharing and Finance has the lowest standard deviation of 1.1%. In
data. To make automatic decisions on users’ behalf, an AI the case of Entertainment, Web browsing tool gets the least
agent needs to understand the factors that influence users’ always share permission preferences, and in the case of
permission decisions. We dive into several factors. Travel, Cloud Drive gets the least always share permission
preferences. Our investigation reveals that, in the context
Communication Context/Domain. Significant prior re-
of Travel, Cloud Drive is always used to access users’
search has identified the “context” of the communication
travel documents (e.g., passport scans, visa records). While
as a key factor in user decision-making around data shar-
participants allow access to these documents as they are
ing [2], [8], [10], [36], [56]. Table 1 presents the eight
essential for some travel tasks, they do not prefer AI agents
contexts (which we refer to as domains) for which we
to automatically retrieve and share such sensitive data.
record user preferences. We observe considerable variation
in participants’ permission preferences in always share and For one-time sharing, Entertainment and Finance have
never share permissions across different domains. For data thehighestandloweststandarddeviationof9.8%and0.2%,
sharing, we note that participants express the most always respectively. For never sharing and one-time non sharing
share permissions (i.e., 55.6%) for the Entertainment do- permissionpreferences,thestandarddeviationbetweentools
mainandtheleastpermissions(i.e.,22.2%)fortheFinance across domains remains mostly stable, and does not exceed
domain.Fortheneversharepermissions,theinverseistrue, 3.9% for never sharing and 2.8% for one-time non sharing.
i.e., participants express their preferences to not share data These results indicate that permission preferences can
7

--- 第 8 页 ---

GT PA Always Yes(Once) No(Once) Never n1.0
NNeecceessssaarryy NUencneescseasrsyary 4311..81%% 4376..10%% 145..92%% 167..17%% ctio0.8 ESnmtaerrtt aHino.me
UUnnnneecceessssaarryy NUencneescseasrsyary 3106..26%% 4281..20%% 1221..01%% 491.5.4%% nt fra0.6 WHeoarklt&h&PrFoitdnu.c.
TABLE 3: Participant permission prefs. for perceived nec- pa0.4 Social
ci Shopping
essary/unnecessarydata(PA),alongwithgroundtruth(GT). rti0.2 Finance
a
P Travel
0.0
0.0 0.1 0.2 0.3 0.4 0.5
varyevenwithindomains,andthusgranularcontextconsid- Permission rate standard deviation
erationisnecessaryforpermissioninferences,i.e.,consider- Figure 3: Distribution of standard deviation (SD) in par-
ing coarse context categories for understanding permission ticipant permission preferences within domains. SD of 0
preferencesacrossusersmaybeinsufficient.Ourpermission implies that participants choose the same preference within
prediction model (in Section 5.1.3) thus encodes contextual adomain,0.1impliesthereare2or3datatypesthatdonot
information at a finer granularity. align with other data types within a domain.
Differences When Users Align and Misalign With the
AI Agent. As we ask participants in our user study to
select the necessary data that the AI agent will need to 4.2.5. Individual participants’ permission preferences
address a query, we develop a notion of participants devel- within a domain are often consistent; however, for some
oping an understanding of the AI agents’ execution. This privacy-conscious participants, there is a high variance
setup allows us to analyze how user permission preferences in their permission decisions. For an AI assistant to learn
may vary when they have some understanding of the AI users’ permission preferences and apply them in various
agent’s execution. Table 3 presents participants’ permission situations, an important criterion is that users’ decision-
preferences for perceived necessary and unnecessary data, making is predictable and consistent. To that end, we an-
alongwiththegroundtruth(asdeterminedinSection4.1.2). alyze the consistency in participants’ permission decisions.
We observe that when participant perceptions of necessary From Figure 3, we note that within a domain, participants’
and unnecessary data align with the ground truth (1st and permission decisions are often consistent. The standard de-
4th row in Table 3), the rate of appropriate permission viation is lowest for the Entertainment and Smart Home
decisions(allowingnecessarydataanddenyingunnecessary domains and highest for the Travel, Finance, and Shopping
data) is high. In such cases, participants are more likely to domains.Notably,manyparticipantshaveentirelyconsistent
express appropriate permissions, with an 88.9% and 62.5% data-sharing preferences within specific domains. There are
correct decision rate for sharing and not sharing. We also 84.5%-85.6%ofparticipantswhoarewillingtoeitherallow
note that, even when participants believe certain necessary ordenyalldatapermissionfortheEntertainmentandSmart
data is unnecessary (2nd row in Table 3), they are still Home domains, 49.2%-56.4% for the Work & Productivity,
able to provide appropriate permissions, likely because data Health & Fitness, and Social Domains, and 32.1%-44.4%
selection for specific queries provides more context for for the Shopping, Finance, and Travel domains.
making an informed decision. Conversely, when the agent Additionally, we notice that some participants have rel-
makesincorrectdecisions,i.e.,presentsunnecessarydataas atively high intra-domain preference variance. We therefore
necessary (3rd row in Table 3), 78.2% of times users share analyze the top 10 participants with high intra-domain pref-
data with the AI agent, which implies that many users tend erence standard deviation and find that the wide variation
to believe the AI agent even when they are incorrect. for these participants can be traced to several key personal
attributes, particularly AI trust levels, privacy concerns, and
4.2.4. Participant demographics information (e.g., age) AI familiarity. Specifically, 8 participants report low to
and self-reported metrics (e.g., privacy consciousness) moderate trust in AI, even though 7 participants use AI
correlate with their permission decision preferences. frequently, and all rate privacy as very important. For these
Next, we analyze participants’ demographic information participants, the sensitivity of data types seems to matter
(i.e.,age,education,andsex)andself-reportedmetrics(i.e., when they make permission decisions. For example, within
AI familiarity, AI usage frequency, AI trust, and privacy theFinancedomain,participantsoftendenyaccesstohighly
consciousness) to analyze how these factors influence par- sensitivepersonaldatasuchasSSNs,bankaccountnumbers,
ticipants’ permission preferences. As for demographics, we and online account credentials, while allowing access to
observethatthetendencytogiveAlwaysAllowdatasharing relativelylesssensitivedata,suchastaxfilingstatus,savings
permissions decreases with participant age, correlating with goals, or payment notes. Such a distinction leads to varied
generationaldifferencesinprivacyattitudesasalsoobserved preferences for participants even within a single domain.
by prior work [2]. We observe that there is no substantial
differenceinthepermissionpreferencesofmaleandfemale 4.2.6. Participants’ permission preferences tend to fol-
participants. As for self-reported metrics, higher AI famil- low similar patterns, presenting an opportunity to pre-
iarityandusagecorrelatewithincreaseddata-sharingprefer- dict individual user preferences by leveraging insights
ences.Weprovideamoredetailedanalysisofdemographics fromotherusers.Inareal-worlddeployment,weanticipate
and self-reported metrics in Appendix A. that users will encounter a range of scenarios that may not
8

--- 第 9 页 ---

on1.00 Top10highstd.dev.datatypes Top10lowstd.dev.datatypes
cti0.75 73.4% Datatype Std.dev. Datatype Std.dev.
a
nt fr0.50 46.3% AEmccpoluonytmCenretdDenettiaaillss 00..449988 MHoubsbicieLsiasntedniInngteHreissttsory 00..112758
pa0.25 DriverLicenseNumber 0.489 RelationshopPreferences 0.212
rtici0.000.0 80.9.1% 0.2 0.3 0.4 0.5 TParasvspelorIttiDneorcauryments 00..448856 TFeitsnte/DssiaGgnooalsticResults 00..224444
Pa Permission rate standard deviation InvestmentInformation 0.483 MoviePreferences 0.246
PaymentMethodDetails 0.482 PersonalBiography 0.248
Figure 4: Distribution of SD in participant permission pref- BankAccountDetails 0.478 TravelPreferences 0.251
erences across domains. SD of 0.1 means that preferences SocialSecurityNumber 0.475 AccommodationDetails 0.255
SenderEmailAddress 0.461 TravelDestination 0.259
are mostly consistent, and SD of 0.2 means that only 1 or
2 domains have a noticeable difference with others. TABLE 4: Data types with high and low standard deviation
in participants’ data permission preferences.
n
o1.00
cti
air fra00..5705 69.7% tshheowhnigihnesTtabalned4,lohwigehs-tvavrairainacbeilittyypeascrlioksesApcacrotiucniptaCnrtse.deAns-
p 60%
nt 0.25 tials,EmploymentDetails,andDriverLicenseNumber(with
a
cip0.00 standard deviations of 0.49) indicate strong disagreement—
Parti 0.0 0.2Permissio0n. 4Jaccard sim0i.l6arity score0.8 1.0 ltiykpeelsydsuucehtoatsheMseunsiscitiLviitsyteonfindgataH.iIsntocroyntarnadst,Hloowb-bviaersiaanncde
Figure 5: Jaccard similarity for participant pairs within Interests (with standard deviations as low as 0.125) reflect
groups.Wegroupparticipantsbythequerysettheyanswer. broadagreement,possiblybecausetheyareseenaslow-risk
and frequently shared in everyday contexts. These patterns
suggest that preferences for some data types are relatively
arise during the training of a personal permission assistant. predictable, while others show diverse responses and are
Forexample,tonotfatiguetheusers,adeployedpermission more challenging to predict.
assistant may record a user’s preferences in one, two, or
a handful of domains, but users’ day to day questions
5. Predicting User Permission Preferences
may span more domains. Thus, we seek to understand how
transferable users’ permission decision making patterns are
In this section, we explore the feasibility of predicting
across contexts. Figure 4 presents the standard deviation of
users’permissiondecisionsbyleveragingdatafromouruser
permissionallowancerateacrossdomainsforusers(Figure9
study.Weexplorethepotentialofusingindividualuserdata
inAppendixBshowsthedistributionofaveragepermission
as well as leveraging data from other users in developing
allowance rates across domains for users.). We note that
an accurate permission prediction model. We also explore
46.3% of participants have a standard deviation below 0.1,
the role of various factors, such as the domains/contexts,
which means their permission is mostly consistent for all
individual data types, and variance in user permission pref-
the tested domains; 73.4% of participants have a standard
erences, in the accuracy of permission prediction.
deviation below 0.2, which means only one or two domains
We anticipate a sustained progression in permission
have a noticeable difference with other domains. Moreover,
modeling for AI agents over the next several years, during
we find that 8.9% of participants maintain consistent per-
which a wide variety of models will be explored. In this
mission preferences across all tested domains. We also find
paper, we focus more on the implications from our user
that over 35.5% of participants (not explicitly represented
studyresultsandpresentourpermissionassistantasaproof
in Figure 4) show zero preference variance in at least half
of concept and/or initial exploration of the design space,
of the domains they interact with, suggesting that they tend
rather than a definitive solution.
to grant similar permissions across different contexts.
We also explore the consistency in permission prefer-
ences for data types across participants, as it can allow 5.1. Designing a Permission Prediction Model
us to learn preferences from groups of users with similar
preferencesandapplythemtootherusers.Figure5presents OurfindingsinSection4.2indicatethatalthoughvarious
theJaccardsimilaritybetweenparticipants’permissionpref- factorsinfluenceparticipants’permissiondecisions,thesein-
erences on the same data permission requests. We note that fluences ultimately result in relatively consistent participant
participants’ permission preferences for the same permis- permission decisions. Thus, we investigate whether users’
sion requests are not unique and in fact, resemble many prior permission decision history, their demographics, and
other users. For example, 69.7% of participant pairs (two other self-reported attributes can be used to predict their
participants answering the same permission requests) have future permissions decisions.
similardatasharingpreferencesfor60%ormoredatatypes. Akeychallengeisthatwepossesslimiteddataonusers,
To further understand user variance and consistency in data which may make it challenging to train a classifier that
permissionpreferences,weanalyzewhichdatatypesexhibit attains a high accuracy. To this end, we explore a hybrid
9

--- 第 10 页 ---

machine learning framework that relies on LLM-based in- that the user would grant the permission. This score is then
context learning [57] and collaborative filtering [29]. We converted into a predicted label (i.e., allow or deny) by ap-
relyonLLM-basedin-contextlearningbecauseitcanattain plying a threshold. The prediction threshold is configurable
ahighaccuracyevenwithahandfulofexamples(i.e.,“few- forcollaborativefiltering,whichwecurrentlysettoequalize
shot”) [19], [57]. We rely on collaborative filtering because FPR and FNR (Appendix C provides more details).
itallowsustolearnfromotheruserswithsimilarpermission
decision history [29], [30], thus complementing the limited 5.1.3. Hybrid model using in-context learning and col-
permission history we possess on individual users. laborative filtering. We next explore combining the in-
context learning and collaborative filtering models to incor-
5.1.1. In-context learning. As we observe in Section 4.2.5 porate learning from both: (1) individual user preferences,
and 4.2.6, participants’ permission decisions remain consis- which in-context learning incorporates, and (2) similar user
tent within a domain and are transferable across domains, preferences, which collaborative filtering incorporates. We
we first attempt to learn preferences only from users’ own combine these models by extending our in-context learning
permission decisions. To that end, we design an LLM- framework (described in Section 5.1.1). Specifically, we in-
based in-context learning framework using OpenAI’s o3- cludetheresultsfromacollaborativefilteringmodelastex-
mini reasoning-based LLM. To condition the LLM, we rely tualdescriptionsfortheLLM.Forexample,arecommended
on role-based prompting, user demographics, other self- permissiondecisionisrepresentedasfollows:<Query:Can
reported information, and user permission history. you retrieve my tax filing details from last year?; Tool:
Since our goal is to design an assistant that will as- TaxManagement;DataType:SSN;Decision:Deny>.Since
sist users in their permission decision making, we adopt preferences for different participants can vary within do-
permission assistant as a role for the LLM. As for the mains (as shown in Section 4.2.3), such representation at
demographics we include, users’ age range, education, and the query level (i.e., finer granularity) allows us to naturally
gender. For self-reported information, we mainly consider capture consistent preferences for individual participants.
users’AIfamiliarity(e.g.,usagefrequency,usagepurposes) While integrating collaborative filtering results, we only
and privacy consciousness (e.g., value to privacy, trust in consider high-confidence permission predictions. We con-
AI). User permission history includes the query, data type, sider high-confidence predictions to be the ones where both
and the name of the tool or AI agent requesting the data, the false positive rate (FPR) and false negative rate (FNR)
along with the user’s permission decision. As LLMs take are below 5%, which covers 35.0% of the permission re-
natural language input, we encode these features as natural quests. Note that in our testing, we tried including all CF
language instructions. For example, demographic informa- predictions(i.e.,both high- andlow-confidencepredictions)
tion is encoded as follows: a male in the 45–54 age group in our hybrid model, but that deteriorates model accuracy.
with a bachelor’s degree. As an output, we condition the Note that our collaborative filtering alone is incapable
LLM to provide a prediction label along with a confidence of making predictions on previously unseen data, and our
score, which prior research has shown to be reliable [62]. in-context learning model only used data from individual
users. Our hybrid model allows collaborative filtering and
5.1.2. Collaborative filtering. Motivated by our observa- in-contextlearningtocomplementeachother.Asaresult,it
tion in Section 4.2.6, where groups of users exhibit similar yields a personalized predictor for each user (i.e., as many
permission-granting behaviors across domains and scenar- personalized instances as users), composed of shared user-
ios, we explore collaborative filtering in predicting user specific CF parameters as well as an in-context learning
preferences.Collaborativefilteringiswidelyadoptedinrec- model trained on individual user data.
ommendation systems due to its effectiveness in capturing
implicit user-user similarities based on commonalities in 5.2. Evaluation
prioruserpreferences[29],[30].Wemodeluserpreferences
as anadjacency matrix, wherecolumns representdata types 5.2.1. Datasets and metrics. We construct our training
acrosscontexts,rowsrepresentusers,andvaluesineachcell and testing datasets using responses collected from the
represent either positive (sharing) or negative (non-sharing) user study to evaluate the effectiveness of our permission
permissionpreference.Sinceusersdonotprovidetheirpref- prediction model. We only consider decisions marked as
erencesonalldatatypes,severalofthecellvaluesareempty, yes, always share or no, never share as ground truth for
andourclassificationtaskistopredictthevaluesofthecells. sharing and non-sharing data. Additionally, we exclude par-
We rely on model-based collaborative filtering, which uses ticipantswhospecifiedalways/neversharingpreferencesfor
a light graph convolution network (LightGCN) [29]. fewerthanfivescenarios/queries.Afterfiltering,ourdataset
LightGCN constructs a graph where users and permis- contains 7,563 permission decisions from 181 participants,
sion requests (with context) are nodes, and edges represent including 5,244 allowance and 2,319 denial responses. We
observed permission preferences. It learns embeddings for define a positive outcome as permission being granted and
usersandpermissionrequestsbyiterativelyaveraginginfor- a negative outcome as permission being denied. For model
mation from their connected neighbors. The final prediction training and evaluation, we perform 5-fold cross-validation.
scoreiscomputedusingthedotproductbetweenauserand Specifically, the user-answered questions are split into five
a permission request embedding, indicating the likelihood folds; in each iteration, we train on four folds and test on
10

--- 第 11 页 ---

1.00 1.00 1.00 1.00
s
Metric Value000...257505 APRFFFC1PNrecoeRccvRcuaeirslrlaaiocgnye Metric value000...257505 APRFFFC1PNrecoeRccvRcuaeirslrlaaiocgnye Metric value000...257505 APRFFFC1PNrecoeRccvRcuaeirslrlaaiocgnye action of user000...257505 APRFFF1PNreceR ccRscuacirslolairoceny
Fr
0.00 0.00 0.00 0.00
0.00 1.00 2.00 3.00 4.00 0.75 0.80 0.85 0.90 0.95 0.75 0.80 0.85 0.90 0.95 0.00 0.25 0.50 0.75 1.00
Confidence threshold Confidence threshold Confidence threshold Metric value
(a) CF (b) IC (c) IC & CF hybrid
Figure 6: Distribution of classification metrics. For IC & hybrid models, the confidence Figure 7: Distribution of hy-
score is reported by the LLM. For CF, the confidence score is the difference b/w prediction brid model metrics over the
thresholds that reduce FPR and FNR, with larger values implying higher confidence. fraction of users.
Metric CF(%) IC(%) IC&CF(%) Hist.ratio 0% 25% 50% 75% 100%
Accuracy 83.3±1.4 84.4±0.7 85.1±0.4 #Queries 0 1-4 2-8 3-12 4-16
Precision 91.9±0.5 91.5±1.1 92.8±0.7
Accuracy(%) 66.9±1.6 77.7±1.3 82.1±0.7 83.7±0.8 85.1±0.4
Recall 83.3±1.4 85.4±1.5 85.2±1.2
Precision(%) 83.0±1.4 87.6±1.2 90.0±0.9 91.1±1.0 92.8±0.7
F1score 87.4±1.0 88.3±0.7 88.8±0.5
Recall(%) 65.7±0.7 79.1±1.5 83.6±0.6 84.7±0.7 85.2±1.2
FPR 16.7±1.5 17.9±2.3 15.2±2.1
F1score(%) 73.4±0.9 83.1±1.0 86.6±0.4 87.8±0.7 88.8±0.5
FNR 16.7±1.4 14.6±1.5 14.8±1.2
FPR(%) 30.6±3.8 25.4±3.2 21.2±2.8 18.7±2.5 15.2±2.1
TABLE 5: Classification metrics with full coverage across FNR(%) 34.3±0.7 20.9±1.5 16.4±0.6 15.3±0.7 14.8±1.2
different model configurations.
TABLE 6: Impact of permission history on classification.
the remaining fold. This is repeated five times, so every
individual model configurations on all metrics, except for
questionistestedexactlyonceandisneverusedfortraining
recall and FNR. Notably, the FPR decreases by 2.7% when
and testing in the same iteration. This setup allows us to
recommendations from the CF model are taken into ac-
evaluate the models on unseen data.
count. These results indicate that incorporating contextual
examples from the CF model enhances prediction accuracy.
5.2.2. Classification confidence threshold. We start by However, as we note in Section 5.1.3 that these gains
comparing model configurations: (1) a model trained on strongly correlate with the quality of CF predictions.
individual user data using in-context (IC) learning, (2) a
As permission assistants will make predictions to share
collaborative filtering (CF) model trained on the entire
dataonbehalfofusers,ahighprecisionandlowerFPRmay
permission-granting history of all users, and (3) a hybrid
be desired, which requires a compromise on model cover-
of in-context learning and collaborative filtering. Figure 6
age. Thus, in practice, the users may need to be involved in
presents the distribution of classification metrics for various
the loop for making data permission decisions (elaborated
model configurations. Overall, we attain the highest accu-
in Section 7). Moreover, as we find (from Section 4.2.5)
racy (95.8%) for CF with a confidence threshold of 3.17.
that usersexpress varying privacy concernsacross domains,
However,wecompromiseoncoverage7 as weareonlyable
custom confidence thresholds could be configured at the
to make predictions on 27.3% of the data. For IC, we attain
granularity of individual users and/or domains. We also
an accuracy of 89.9% with a confidence threshold of 0.90,
find (from Section 4.2.6) that certain data types show a
and a coverage of only 18.8%. For the hybrid model of IC
high degree of variance across users, so they may also be
and CF, we attain an accuracy of 94.4% with a confidence
excluded from predictions to avoid mistakes.
threshold of 0.91, and a coverage of 25.9%. These trends
indicate that, as the confidence threshold gets stricter, it
5.2.3. Impact of permission decision history. In a real-
results in precise predictions but reduces the coverage.
world setting, users may not be expected to provide per-
Classification Accuracy. To make classification decisions mission preferences for a large number of data types. Thus,
for all data, i.e., for 100% coverage, across all model weexploretherelationbetweennumberoftrainingsamples
configurations,weareabletoachieveanaccuracyof83.3%, available per user and accuracy of our permission assistant.
84.4%, and 85.1%, for CF, IC, and the hybrid IC and CF To assess the impact of permission decision history on
model. Table 5 presents other classification metrics for full prediction accuracy, we incorporate different ratios of the
data coverage. We note that the hybrid model outperforms permission decision history into each user’s model context
and evaluate the corresponding performance. Table 6 shows
7.Wedefinecoverageasthefractionofpermissionrequestsforwhich
the metrics for our hybrid model under various ratios of
the model can recommend an action with confidence above a specified
permission decision history. We note that even without any
threshold.Lowercoverage,however,alsomeansthatusersmustmakemore
decisionsmanually,andonlyhigh-confidencecasesareautomated. permission history, the model still infers users’ preferences
11

--- 第 12 页 ---

by leveraging users’ demographic information, AI usage vs. 4.14), these differences are not statistically significant
experience, and privacy concerns. We also observe that the (Mann-Whitney U test p-values range from 0.07 to 0.95).
transitionfromnopermissionhistorytoevenasmallamount We then examine whether there are significant differences
(i.e., permission history on 1–4 queries) yields significant in the number of permission decision history entries and
improvements (i.e., more than 10.8% increase in accuracy). the number of collaborative filtering recommendations (as
Overall, our results indicate that an increase in per- we observe them to be key factors in improving prediction
mission decision history (i.e., an increase in training data) accuracy in Section 5.2.2 and 5.2.3). We observe notable
results in improved prediction accuracy. This continuous differences in the average number of permission history
learningcapabilitycanfacilitatereal-worlddeployment,i.e., queries between the two groups (11.69 for high-accuracy
as users naturally generate more permission history over users vs. 9.38 for low-accuracy users, U-test p = 0.01), as
time, the model can iteratively learn from them, leading to well as in the number of CF recommendations (3.96 vs.
increasingly accurate permission predictions. 2.90, U-test p=0.07).
Next, we analyze the standard deviation in permission
5.3. Result Analysis allowance rates across domains for both high-accuracy and
low-accuracy users. We note that high-accuracy users tend
to have lower variability in their permission preferences
5.3.1. Per-user performance analysis. As each user has
across domains, with a larger proportion of users exhibiting
their personalized model for predicting permission deci-
smallerstandarddeviationvalues,indicatingmoreconsistent
sions, we analyze how these models perform on permission
preferences. In contrast, low-accuracy users show greater
decision predictions for each user. Specifically, we compute
variability in their permission preferences across domains.
per-usermetricsusingthepredictionresultsfromeachuser’s
For instance, 30.6% of high-accuracy users fall at the lower
dedicated hybrid IC and CF model, and attempt to identify
end of the standard deviation range (≤ 0.04), compared to
the factors that contribute to lower or higher accuracy.
just 11.1% of low-accuracy users. Similarly, at the upper
Overall Trends. Figure 7 presents the distribution for per-
end (≥ 0.27), 22.2% of low-accuracy users exceed this
formance metrics computed over individual users. Notably,
standard deviation threshold (0.27), while only 13.9% of
35.4% of users achieve accuracy greater than 90%, and
high-accuracy users do. We observe that high standard de-
12.7%ofusersevenachieveperfectaccuracy,meaningevery
viation in users’ permission decisions may make it more
permission decision predicted by their dedicated model is
challenging to accurately predict their permission prefer-
correct. For precision, 70.4% of users reach values greater
ences.Forinstance,predictingthepreferencesofauserwith
than or equal to 90%, suggesting that these models are
astandarddeviationof0.43yieldssubparperformance,with
reliable in correctly predicting data sharing (i.e., predicting
an accuracy of 49.0%.
positive cases). In contrast, 34.8% of users achieve recall
A closer look at this user’s prediction results reveals
valuesof90%orhigher,indicatingthatmodelsaregenerally
the impact of such variance: although the user is generally
conservative in making automated data sharing predictions.
willing to share data in domains like Smart Home, Travel,
The error metrics further elaborate on model perfor- and Health & Fitness, they tend to deny data sharing in
mance.ThedistributionforFPRshowsthat48.2%ofusers’ Social, Finance, and Shopping. We surmise that when the
falsepermissiongrants(i.e.,FPR)areatorbelow10%(with
model sees more permission history from the first group
0 false permission grants for 37.2% of users). Meanwhile,
of domains, it tends to infer a higher allowance rate even
34.8% of users have an FNR of 10% or less, indicating
for the latter group, and vice versa, during five-fold cross-
that more than a third of the users’ models are able to
validation. This suggests that for users with high variance
constrain the number of false permission denials. Overall,
in their permission preferences, patterns learned from some
the personalized models show promising performance in
domains may not transfer well to others, resulting in re-
predicting individual users’ permission decisions. In the
duced prediction performance. In contrast, when users have
remainder of this section, we analyze the errors to gather
moreconsistentpermissionpreferencesacrossdomains(i.e.,
insights for further improving the models.
lower variance), the model finds it easier to transfer learned
Detailed Analysis. We perform an in-depth analysis of the patterns. For example, a high-accuracy user with a standard
bottom 20% (36) users with the lowest accuracy by com- deviation of just 0.01 achieves an accuracy of 98.0%, recall
paring this group to the top 20% of users with the highest of 100.0%, and precision of 97.8%. We note that it is
accuracy. There are significant differences in the average a key challenge to make predictions of users’ permission
performancemetricsbetweenthesetwogroups,forinstance, preferences when there is a high degree of variance in their
average accuracy (97.6% vs. 69.7%), recall (98.2% vs. prior permission decisions.
62.3%), and precision (98.0% vs. 69.2%). Next, we investi- Finally,weinvestigatetheimpactofCFrecommendation
gatethefactorsthatmaycontributetothesedifferences.We accuracyonthetwousergroups.Wefindthathigh-accuracy
first examine whether users’ self-reported attributes show usershaveCFrecommendationswithanaccuracyof99.5%,
significantdiscrepancies.Whilesometrendsareobservable, compared to 89.5% CF recommendation accuracy for low-
for example, high-accuracy users report slightly lower AI accuracy users. Additionally, when CF recommendations
trust levels (2.39 vs. 2.81 on average), higher AI usage are correct, the model consistently follows them, showing
frequency (3.19 vs. 3.08), and lower privacy concerns (4.06 100%alignmentforlow-accuracyusersand99.8%forhigh-
12

--- 第 13 页 ---

accuracyusers.However,whenCFrecommendationsarein- cies, including SMS (76.4%), Cloud Drive (76.4%), Slack
correct,themodelstilltendstofollowthem.Thishighlights (79.3%), Lighting Control (80.0%), and Banking (80.3%).
theimportanceoffilteringoutunreliableCFresultstoavoid Among these tools, Cloud Drive (FNR of 31.4%), Banking
misleading the model’s final decisions. For CF predictions (27.3%), and Lighting Control (20.9%) suffer from high
to improve, it is crucial that users can be grouped with FNRs, reflecting the model’s tendency toward caution.
others who share similar preferences. Thus, as more data is
Performance by Data Type. We observe substantial dif-
collected—particularly from a larger and more diverse user
ferences while analyzing the model’s accuracy across data
base—the quality of CF predictions naturally improves.
types. For instance, some data types yield extremely high
accuracies, even reaching 100%, such as Fitness Goal and
5.3.2. Contextualperformanceanalysis.Next,weanalyze HobbiesandInterests.Otherdatatypeswithsimilarlystrong
howwellthemodelpredictsuserpreferencesunderdifferent results are Relationship Preferences (98.1%), Personal Bi-
contexts, specifically across domains, tools, and data types. ography (96.2%), Travel Destination (95.5%), and Accom-
Our goal is to examine whether there are significant differ- modation Details (95.4%). It is worth noting that these data
ences in accuracy, FPR, and FNR among these contextual typesalsorankamongthosewiththelowestpreferencevari-
factors, thereby revealing patterns in when and where the anceamongparticipants,asshowninTable4.Thissupports
model tends to be over- or under-permissive. our hypothesis that data types with more consistent user
Performance by Domain. When analyzing performance preferences are easier for the model to predict accurately.
across different domains, we find that the model performs On the other hand, data types like Payment Method De-
slightly better at predicting permission requests in certain tail (59.1%), Employment Details (63.6%), Driver License
domains. Notably, Work & Productivity (87.4% accuracy), Number (71.4%), and Travel Itinerary (75.9%) show some
Health & Fitness (87.1%), and Entertainment (86.8%) ex- of the lowest accuracies. These data types also have the
hibit relatively high accuracies. In contrast, domains such highest variance in user preferences, according to Table 4.
asShopping(84.3%accuracy),Finance(81.5%),andSmart In other words, data types with more diverse permission
Home (80.1%) show lower prediction accuracies. preferences tend to be more difficult for the model to
predict. Other low-accuracy data types often involve highly
We find that these results correlate with the variance of
sensitiveinformationaswell,suchasPersonalIdentification
permission decisions in domains (Figure 3 plots standard
Numbers (61.9%) and Travel History (63.2%). These types
deviation across domains). Specifically, the high-accuracy
ofdataalsotendtohavehigherFNRs.Forinstance,Payment
domains have lower variance in participants’ permission
Method Details has an FNR of 87.5%, Personal Identifi-
preferences and low-accuracy domains have the highest
cation Numbers has 72.2%, and Driver License Number
variance in participants’ permission preferences. A closer
examination of the Smart Home results reveals a high FNR has 66.7%. These are among the most under-permissioned
data types. Only a few show relatively high FPRs, such
of23.8%,indicatingaconservativebiasinwhichthemodel
as Employment Details and Travel History, both at 28.6%.
tends to reject permissions that users might have accepted
in this domain. Notably, Finance has the highest FNR of Theseresultssuggestthatthemodeloftenmisclassifiesthese
sensitive data types as unacceptable to share, even when
all domains, at 28.3%, which reflects the model’s cautious
users may permit it. This reflects challenges in accurately
behavior in this particularly sensitive domain. By contrast,
the Social domain exhibits the highest FPR, at 23.3%, sug- capturing context-sensitive boundaries for personally iden-
tifiable or other sensitive information.
gesting that the model is more likely to approve permission
requests in less sensitive domains.
Overall, we note that while variance within domains 6. Discussion
degrades accuracy, the degradation is not substantial, espe-
ciallyascomparedtothedegradationinaccuracywithvari-
Reinforcing and Expanding Prior Knowledge. Our find-
ance within the user’s permission decisions (Section 5.3.1).
ings on user permissions for AI agents confirm established
Performance by Tool. At the tool level, disparities be- access control principles from traditional systems while
comeevenmoreapparent.Toolswiththehighestaccuracies alsorevealingagent-specificdynamics.Consistentwithprior
include Calendar (92.0%), The Movie Database (89.0%), work on mobile applications and voice assistants [2], [43],
Tinder (88.8%), Hospital Booking (88.6%), and Weather [49], [63], [70], our results show that over-permissioning
(88.4%). These are also the tools for which participants remains a key challenge, and that user decisions are highly
tend to grant more permissions. For example, as shown dependent on context, such as data sensitivity and the in-
in Table 2, Calendar appears in multiple domains such as formation’s recipient. However, the autonomous nature of
Travel, Work & Productivity, and Health & Fitness, and AI agents fundamentally alters the control dynamic. We
it ranks among the top tools with the most Always allow find that permission granting becomes a continuous process
permissions.Asimilarpatternholdsfortheotheraforemen- of trust evaluation: when an agent makes a mistake, users
tioned tools, suggesting that the model predicts permissions aresignificantlylesswillingtograntpermissions,indicating
moreaccuratelywhenthereismoredataavailable,andusers that agent performance is a new, critical factor influencing
have consistent preferences for specific tools. privilege. Most critically for our work, we find that despite
In contrast, some tools have noticeably lower accura- these complexities, individual preferences exhibit a high
13

--- 第 14 页 ---

degree of consistency within specific contexts and across effective AI permission assistant can substantially reduce
users.Thispredictablepatternisthekeyinsightthatenables the number of decisions that users must be asked to make
our permission-prediction system, demonstrating that foun- or evaluate directly, paving the way for a usable and secure
dational privacy norms persist while the agent’s autonomy permission management system.
introduces new factors for permission management.
7. Conclusion
Model Robustness and Limitations. As agents interface
with data and resources from potentially untrustworthy en-
In this paper, we explored the capabilities and limits of
tities, it is crucial that the permission inference is robust
automating permission management in AI agents. Through
against such attacks. When users communicate with AI
auserstudy,wefoundthatlong-standingchallengessuchas
agents using natural language instructions, they often re-
over-permissioningpersistinagenticsystems,alongsidenew
veal implicit or explicit preferences about what the agent
issuesuniquetothem.Notably,whenagentsmakemistakes,
is allowed to do. Such information can complement the
users become less willing to grant permissions, indicat-
executioncontextandpastuserbehaviorinmakingaccurate
ing that agent performance is a critical factor influencing
predictions. In fact, prior work has used such information
privilege decisions. We also found that permission choices
to predict the expected control and data flow of AI agents
are strongly shaped by communication context, yet remain
and detect anomalies [16], [79]. Similarly, prior work has
consistentwithinthatcontextacrossusergroups.Leveraging
proposed AI agent architectures that, by design, limit the
these insights, we developed a permission prediction model
flow of information between system modules [8], [79] and
that combines collaborative filtering with LLM in-context
reliably control the generation of text in LLMs [52]. We
learning,achieving94.4%accuracyforhigh-confidencepre-
believe that such approaches can be extended to support
dictions. Even without prior permission history, the model
robust permission management in AI agents.
reached66.9%accuracy,withjust1–4samplesimprovingit
That said, natural language interactions can be ambigu- by10.8%.Ourfindingsdemonstratethepotentialoflearning
ous, and this remains an open problem for LLMs [45]. user preferences to automate many permission decisions.
Our in-context learning model can be affected by this am- However, challenges remain in enforcing predictions, im-
biguity and may make mistakes when processing unclear provingrobustness,anddesigningusableinterfaces.Overall,
instructions.However,becausewealsorelyoncollaborative this work advances automated permission management and
filteringoverdeterministicdatatypes,thesysteminherently outlines key directions toward its practical deployment.
offers some resilience to such ambiguity for previously
seen data types. Once a request is mapped to a canonical Acknowledgment
label, collaborative filtering predictions become insensitive
to wording. Moreover, taking only high-confidence predic- We thank the reviewers for their valuable feed-
tions and delegating uncertain data access permissions to back. This work was partially supported by NSF (CNS-
users can help mitigate the model’s robustness issues. This 2154930, CNS-2238635), ARO (W911NF-24-1-0155),
approach pairs predictions with clear controls to make and ONR(N000142412663),andgiftsfromMicrosoft.T.Kohno
revokedecisionsandtoprovidefeedback,whichreducesin- was also supported by the McDevitt Chair in Computer
terruptionsforroutinecaseswhileallowinghumanoversight Science, Ethics, and Society at Georgetown University; the
in riskier or uncertain situations. majority of this research was conducted while he was at
the University of Washington. We would also like to thank
Towards Usable Permission Management. Automated
Pardis Emami-Naeini for providing feedback on the user
permission management is a multifaceted problem. Our
study and statistical analysis.
work in this paper focuses on one facet, and other impor-
tant facets, including improving the usability of permission
References
management in AI agents, remain challenges and open
avenues for future work. For example, during early use,
[1] Gpt action authentication. https://platform.openai.com/docs/actions/
the assistant may still need to learn a user’s preferences; a
authentication,2025. Accessed:2025-06-02.
user’spreferencesmaychangeastheircircumstanceschange
[2] Noura Abdi, Xiao Zhan, Kopo M Ramokapane, and Jose Such.
(e.g., a previously trusted entity becomes untrusted); and Privacy norms for smart home personal assistants. In Proceedings
some situations may be fundamentally difficult to predict. ofthe2021CHIconferenceonhumanfactorsincomputingsystems,
Thus, a full-fledged permission system will need to include pages1–14,2021.
not only a permission prediction module, but also UI/UX [3] Hazim Almuhimedi, Florian Schaub, Norman Sadeh, Idris Adjerid,
AlessandroAcquisti,JoshuaGluck,LorrieFaithCranor,andYuvraj
for engaging the user directly in cases where the predictive
Agarwal.Yourlocationhasbeenshared5,398times!afieldstudyon
modelhaslowconfidenceormakesamistake—forexample, mobileappprivacynudging.InProceedingsofthe33rdannualACM
UI/UX for users to make explicit permission decisions, to conferenceonhumanfactorsincomputingsystems,pages787–796,
revokepreviously-grantedpermissions,andtogivefeedback 2015.
to the permission assistant. We believe there is rich future [4] Marc Serramia Amoros, William Seymour, Natalia Criado, and
Michael Luck. Predicting privacy preferences for smart devices as
work to be done on how to design such a hybrid system
norms. InThe22ndInternationalConferenceonAutonomousAgents
well.Theimportantcontributionofourworkhereistostart
and Multiagent Systems. International Foundation for Autonomous
this line of inquiry and to demonstrate that it is feasible: an AgentsandMultiagentSystems(IFAAMAS),2023.
14

--- 第 15 页 ---

[5] AndroidDevelopers.Permissionsoverview.https://developer.android. [21] Adrienne Porter Felt, Erika Chin, Steve Hanna, Dawn Song, and
com/guide/topics/permissions/overview, 2023. Accessed: 2025-03- DavidWagner. Androidpermissionsdemystified. InProceedingsof
26. the18thACMconferenceonComputerandcommunicationssecurity,
pages627–638,2011.
[6] Android Open Source Project. Runtime permissions. https://source.
android.com/docs/core/permissions/runtime perms, 2023. Accessed: [22] Adrienne Porter Felt, Elizabeth Ha, Serge Egelman, Ariel Haney,
2025-03-26. ErikaChin,andDavidWagner. Androidpermissions:Userattention,
comprehension, and behavior. In Proceedings of the eighth sympo-
[7] Anthropic. Developingacomputerusemodel,2024. siumonusableprivacyandsecurity,pages1–14,2012.
[8] Eugene Bagdasarian, Ren Yi, Sahra Ghalebikesabi, Peter Kairouz, [23] DorotaFilipczuk,TimBaarslag,EnricoHGerding,andMCSchrae-
Marco Gruteser, Sewoong Oh, Borja Balle, and Daniel Ramage. fel. Automated privacy negotiations with preference uncertainty.
Airgapagent: Protecting privacy-conscious conversational agents. In AutonomousAgentsandMulti-AgentSystems,36(2):49,2022.
Proceedingsofthe2024onACMSIGSACConferenceonComputer
andCommunicationsSecurity,pages3868–3882,2024. [24] Janet Finch. The vignette technique in survey research. Sociology,
21(1):105–114,1987.
[9] Nata˜ MBarbosa,JoonSPark,YaxingYao,andYangWang. “what
[25] GoogleDeepMind. Introducinggemini2.0:ournewaimodelforthe
if?”predictingindividualusers’smarthomeprivacypreferencesand
agenticera,2024.
theirchanges.ProceedingsonPrivacyEnhancingTechnologies,2019.
[26] MarianHarbach,IgorBilogrevic,EnricoBacis,SerenaChen,Ravjit
[10] AdamBarth,AnupamDatta,JohnCMitchell,andHelenNissenbaum.
Uppal, Andy Paicu, Elias Klim, Meggyn Watkins, and Balazs En-
Privacyandcontextualintegrity:Frameworkandapplications.In2006
gedy.Don’tinterruptme–alarge-scalestudyofon-devicepermission
IEEE symposium on security and privacy (S&P’06), pages 15–pp.
promptquietinginchrome.InNDSS.TheInternetSociety,SanDiego,
IEEE,2006.
CA,2024.
[11] Igor Bilogrevic, Balazs Engedy, Judson L Porter III, Nina Taft,
[27] David Harborth and Sebastian Pape. Investigating privacy concerns
Kamila Hasanbega, Andrew Paseltiner, Hwi Kyoung Lee, Edward
related to mobile augmented reality apps–a vignette based online
Jung,MeggynWatkins,PJMcLachlan,etal.”shhh...be{quiet!}”re-
experiment. ComputersinHumanBehavior,122:106833,2021.
ducingtheunwantedinterruptionsofnotificationpermissionprompts
onchrome. In30thUSENIXSecuritySymposium(USENIXSecurity [28] Harrison Chase and the LangChain Team. Langchain. https://www.
21),pages769–784,2021. langchain.com/,2025.
[12] Andre´ Branda˜o, Ricardo Mendes, and Joa˜o P Vilela. Prediction [29] Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang,
of mobile app privacy preferences with user profiles via federated and Meng Wang. Lightgcn: Simplifying and powering graph con-
learning. In Proceedings of the Twelfth ACM Conference on Data volution network for recommendation. In Proceedings of the 43rd
andApplicationSecurityandPrivacy,pages89–100,2022. International ACM SIGIR conference on research and development
inInformationRetrieval,pages639–648,2020.
[13] TomBrown,BenjaminMann,NickRyder,MelanieSubbiah,JaredD
Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, [30] XiangnanHe,LiziLiao,HanwangZhang,LiqiangNie,XiaHu,and
Girish Sastry, Amanda Askell, et al. Language models are few- Tat-SengChua. Neuralcollaborativefiltering. InProceedingsofthe
shot learners. Advances in neural information processing systems, 26th international conference on world wide web, pages 173–182,
33:1877–1901,2020. 2017.
[31] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin
[14] David Chan. So why ask me? are self-report data really that bad?
Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng
In Statistical and methodological myths and urban legends, pages
Feng, Bing Qin, et al. A survey on hallucination in large language
329–356.Routledge,2010.
models:Principles,taxonomy,challenges,andopenquestions. ACM
[15] AnupamDas,MartinDegeling,DanielSmullen,andNormanSadeh. TransactionsonInformationSystems,43(2):1–55,2025.
Personalized privacy assistants for the internet of things: Providing
[32] Umar Iqbal, Pouneh Nikkhah Bahrami, Rahmadi Trimananda, Hao
userswithnoticeandchoice. IEEEPervasiveComputing,17(3):35–
Cui, Alexander Gamero-Garrido, Daniel Dubois, David Choffnes,
46,2018.
AthinaMarkopoulou,FranziskaRoesner,andZubairShafiq. Track-
[16] Edoardo Debenedetti, Ilia Shumailov, Tianqi Fan, Jamie Hayes, ing, profiling, and ad targeting in the alexa echo smart speaker
Nicholas Carlini, Daniel Fabian, Christoph Kern, Chongyang Shi, ecosystem. InACMInternetMeasurementConference(IMC),2023.
AndreasTerzis,andFlorianTrame`r. Defeatingpromptinjectionsby
[33] UmarIqbal,TadayoshiKohno,andFranziskaRoesner. Llmplatform
design. arXivpreprintarXiv:2503.18813,2025.
security: applying a systematic evaluation framework to openai’s
[17] Edoardo Debenedetti, Jie Zhang, Mislav Balunovic´, Luca Beurer- chatgpt plugins. In Proceedings of the AAAI/ACM Conference on
Kellner, Marc Fischer, and Florian Trame`r. Agentdojo: A dynamic AI,Ethics,andSociety,volume7,pages611–623,2024.
environment to evaluate attacks and defenses for llm agents. arXiv [34] UmarIqbal,TadayoshiKohno,andFranziskaRoesner. Llmplatform
preprintarXiv:2406.13352,2024. security: applying a systematic evaluation framework to openai’s
[18] Verena Distler, Matthias Fassl, Hana Habib, Katharina Krombholz, chatgpt plugins. In Proceedings of the AAAI/ACM Conference on
GabrieleLenzini,CarineLallemand,VincentKoenig,andLorrieFaith AI,Ethics,andSociety,volume7,pages611–623,2024.
Cranor. Empirical research methods in usable privacy and security. [35] Jerry Liu and Simon Suo and the LlamaIndex Team. Llamaindex.
InHumanFactorsinPrivacyResearch,pages29–53.SpringerInter- https://www.llamaindex.ai/,2025.
nationalPublishingCham,2023.
[36] Yunhan Jack Jia, Qi Alfred Chen, Shiqi Wang, Amir Rahmati, Ear-
[19] Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Jingyuan Ma, Rui lence Fernandes, Zhuoqing Morley Mao, Atul Prakash, and SJ Un-
Li,HemingXia,JingjingXu,ZhiyongWu,BaobaoChang,etal. A viersity.Contexlot:Towardsprovidingcontextualintegritytoappified
surveyonin-contextlearning.InProceedingsofthe2024Conference iotplatforms. Inndss,volume2,pages2–2.SanDiego,2017.
onEmpiricalMethodsinNaturalLanguageProcessing,pages1107–
[37] Patrick Gage Kelley, Sunny Consolvo,Lorrie Faith Cranor, Jaeyeon
1128,2024.
Jung,NormanSadeh,andDavidWetherall. Aconundrumofpermis-
[20] Steven Englehardt and Arvind Narayanan. Online tracking: A 1- sions:installingapplicationsonanandroidsmartphone. InFinancial
million-site measurement and analysis. In Proceedings of the 2016 Cryptography and Data Security: FC 2012 Workshops, USEC and
ACMSIGSACconferenceoncomputerandcommunicationssecurity, WECSR2012,Kralendijk,Bonaire,March2,2012,RevisedSelected
pages1388–1401,2016. Papers16,pages68–79.Springer,2012.
15

--- 第 16 页 ---

[38] Leona Lassak, Hanna Pu¨schel, Tobias Gostomzyk, and Markus [55] Pardis Emami Naeini, Sruti Bhagavatula, Hana Habib, Martin
Du¨rmuth. Introducingdatatrustees:Avignette-basedstudyapproach Degeling, Lujo Bauer, Lorrie Faith Cranor, and Norman Sadeh.
togetusersintheloop. 2023. Privacyexpectationsandpreferencesinan{IoT}world.InThirteenth
symposiumonusableprivacyandsecurity(SOUPS2017),pages399–
[39] Leona Lassak, Hanna Pu¨schel, Oliver D Reithmaier, Tobias Gos-
412,2017.
tomzyk,andMarkusDu¨rmuth.Balancingprivacyanddatautilization:
Acomparativevignettestudyonuseracceptanceofdatatrusteesin [56] Helen Nissenbaum. Privacy as contextual integrity. Wash. L. Rev.,
germany and the us. In Network and Distributed System Security 79:119,2004.
(NDSS)Symposium,2025.
[57] Harsha Nori, Yin Tat Lee, Sheng Zhang, Dean Carignan, Richard
[40] Josephine Lau, Benjamin Zimmerman, and Florian Schaub. Alexa, Edgar, Nicolo Fusi, Nicholas King, Jonathan Larson, Yuanzhi Li,
areyoulistening?privacyperceptions,concernsandprivacy-seeking Weishung Liu, et al. Can generalist foundation models outcompete
behaviorswithsmartspeakers. ProceedingsoftheACMonhuman- special-purpose tuning? case study in medicine. arXiv preprint
computerinteraction,2(CSCW):1–31,2018. arXiv:2311.16452,2023.
[41] Junlong Li, Fan Zhou, Shichao Sun, Yikai Zhang, Hai Zhao, and [58] OpenAI. Consequential flag. https://platform.openai.com/docs/
PengfeiLiu. Dissectinghumanandllmpreferences. InProceedings actions/production/consequential-flag#consequential-flag,2024.
of the 62nd Annual Meeting of the Association for Computational
Linguistics,pages1790–1811,2024. [59] OpenAI. Introducinggpts. https://openai.com/blog/introducing-gpts,
2024.
[42] XiaotianVivianLi,MaryBethRosson,andJenayRobert.Ascenario-
basedexplorationofexpectedusefulness,privacyconcerns,andadop- [60] OpenAI. Chatgpt. https://chatgpt.com/,2025.
tionlikelihoodoflearninganalytics.InProceedingsoftheninthACM [61] OpenAI. Introducingoperator,2025.
conferenceonlearning@scale,pages48–59,2022.
[62] YudiPawitanandChrisHolmes.Confidenceinthereasoningoflarge
[43] Jialiu Lin, Bin Liu, Norman Sadeh, and Jason I Hong. Modeling languagemodels. HarvardDataScienceReview,7(1),2025.
{Users’} mobile app privacy preferences: Restoring usability in a
sea of permission settings. In 10th Symposium On Usable Privacy [63] Sai Teja Peddinti, Igor Bilogrevic, Nina Taft, Martin Pelikan, U´lfar
andSecurity(SOUPS2014),pages199–212,2014. Erlingsson, Pauline Anthonysamy, and Giles Hogben. Reducing
permission requests in mobile apps. In Proceedings of the internet
[44] Alisa Liu, Zhaofeng Wu, Julian Michael, Alane Suhr, Peter West,
measurementconference,pages259–266,2019.
Alexander Koller, Swabha Swayamdipta, Noah A Smith, and Yejin
Choi. We’re afraid language models aren’t modeling ambiguity. In [64] Prolific. Prolific - recruitment platform for online research. https:
Proceedingsofthe2023ConferenceonEmpiricalMethodsinNatural //www.prolific.com.
LanguageProcessing,pages790–807,2023.
[65] AbbasRazaghpanah,RishabNithyanand,NarseoVallina-Rodriguez,
[45] Alisa Liu, Zhaofeng Wu, Julian Michael, Alane Suhr, Peter West, SrikanthSundaresan,MarkAllman,ChristianKreibich,PhillipaGill,
Alexander Koller, Swabha Swayamdipta, Noah A Smith, and Yejin et al. Apps, trackers, privacy, and regulators: A global study of
Choi. We’re afraid language models aren’t modeling ambiguity. In the mobile tracking ecosystem. In The 25th Annual Network and
Proceedingsofthe2023ConferenceonEmpiricalMethodsinNatural DistributedSystemSecuritySymposium(NDSS2018),2018.
LanguageProcessing,pages790–807,2023.
[66] Harry T Reis and Charles M Judd. Handbook of research methods
[46] Bin Liu, Mads Schaarup Andersen, Florian Schaub, Hazim Al- in social and personality psychology. Cambridge University Press,
muhimedi,ShikunAerinZhang,NormanSadeh,YuvrajAgarwal,and 2000.
Alessandro Acquisti. Follow my recommendations: A personalized
[67] Talia Ringer, Dan Grossman, and Franziska Roesner. Audacious:
privacyassistantformobileapppermissions. InTwelfthsymposium
User-driven access control with unmodified operating systems. In
onusableprivacyandsecurity(SOUPS2016),pages27–41,2016.
Proceedingsofthe2016ACMSIGSACConferenceonComputerand
[47] Gary Liu and Nathan Malkin. Effects of privacy permissions on CommunicationsSecurity,pages204–216,2016.
user choices in voice assistant app stores. Proceedings on Privacy
[68] Franziska Roesner, Tadayoshi Kohno, Alexander Moshchuk, Bryan
EnhancingTechnologies,2022.
Parno,HelenJWang,andCrispinCowan.User-drivenaccesscontrol:
[48] NathanMalkin,SergeEgelman,andDavidWagner. Privacycontrols Rethinkingpermissiongrantinginmodernoperatingsystems.In2012
for always-listening devices. In Proceedings of the New Security IEEE Symposium on Security and Privacy, pages 224–238. IEEE,
ParadigmsWorkshop,pages78–91,2019. 2012.
[49] Nathan Malkin, David Wagner, and Serge Egelman. Runtime per- [69] Jasmin Schwab, Alexander Nussbaum, Anastasia Sergeeva, Florian
missionsforprivacyinproactiveintelligentassistants. InEighteenth Alt,andVerenaDistler. Whatmakesphishingsimulationcampaigns
Symposium on Usable Privacy and Security (SOUPS 2022), pages (un)acceptable?avignetteexperiment. InNetworkandDistributed
633–651,2022. SystemSecuritySymposium,NDSS2025,2025.
[50] ManusAI.Manus-anautonomousartificialintelligenceagent.https: [70] Suranga Seneviratne, Aruna Seneviratne, Prasant Mohapatra, and
//manus.im/. AnirbanMahanti. Yourinstalledappsrevealyourgenderandmore!
[51] RicardoMendes,MarianaCunha,Joa˜oPVilela,andAlastairRBeres- ACMSIGMOBILEMobileComputingandCommunicationsReview,
ford. Enhancing user privacy in mobile devices through prediction 18(3):55–61,2015.
of privacy preferences. In European Symposium on Research in [71] Yashothara Shanmugarasa, Hye-young Paik, Salil S Kanhere, and
ComputerSecurity,pages153–172.Springer,2022. Liming Zhu. Automated privacy preferences for smart home data
[52] MichalMoskal,MadanMusuvathi,andEmreKıcıman.AIController sharing using personal data stores. IEEE Security & Privacy,
Interface. https://github.com/microsoft/aici/,2024. 20(1):12–22,2021.
[53] SaraMotiee,KirstieHawkey,andKonstantinBeznosov.Dowindows [72] HanShao,XiangLi,andGuodiWang.Areyoutired?iam:Tryingto
usersfollowtheprincipleofleastprivilege?investigatinguseraccount understandprivacyfatigueofsocialmediausers. InCHIConference
controlpractices. InProceedingsoftheSixthSymposiumonUsable onHumanFactorsinComputingSystemsExtendedAbstracts,pages
PrivacyandSecurity,pages1–13,2010. 1–7,2022.
[54] RajivMovva,PangWeiKoh,andEmmaPierson. Annotationalign- [73] Andreas Sotirakopoulos, Kirstie Hawkey, and Konstantin Beznosov.
ment:Comparingllmandhumanannotationsofconversationalsafety. Onthechallengesinusablesecuritylabstudies:Lessonslearnedfrom
In Proceedings of the 2024 Conference on Empirical Methods in replicating a study on ssl warnings. In Proceedings of the Seventh
NaturalLanguageProcessing,pages9048–9062,2024. SymposiumonusablePrivacyandSecurity,pages1–18,2011.
16

--- 第 17 页 ---

[74] RahmadiTrimananda,HieuLe,HaoCui,JaniceTranHo,Anastasia > 54 PhD F
Shuba, and Athina Markopoulou. {OVRseen}: Auditing network 40-54 MS
trafficandprivacypoliciesinoculus{VR}.In31stUSENIXsecurity 25-39 BS
symposium(USENIXsecurity22),pages3789–3806,2022. < 25 HS M
[75] Han Wang, An Zhang, Nguyen Duy Tai, Jun Sun, Tat-Seng Chua, 0% 20%40%60%80%100% 0% 20% 40% 60% 80%100% 0% 20% 40% 60% 80%100%
et al. Ali-agent: Assessing llms’ alignment with human values via (a) Age group (b) Education level (c) Sex
agent-basedevaluation. AdvancesinNeuralInformationProcessing
Systems,37:99040–99088,2024. 4 4
[76] Julia Wiesinger, Patrick Marlow, and Vladimir Vuskovic. Agents, 3 3
2025. https://www.kaggle.com/whitepaper-agents.
2 2
[77] PrimalWijesekera,ArjunBaokar,AshkanHosseini,SergeEgelman,
David Wagner, and Konstantin Beznosov. Android permissions re- 1 1
mystified: A field study on contextual integrity. In 24th USENIX
SecuritySymposium(USENIXSecurity15),pages499–514,2015. 0% 20% 40% 60% 80% 100% 0% 20% 40% 60% 80% 100%
(d) AI tool familiarity (e) AI tool usage frequency
[78] Yuhao Wu, Evin Jaff, Ke Yang, Ning Zhang, and Umar Iqbal. An
in-depth investigation of data collection in llm app ecosystems. In
ACMInternetMeasurementConference(IMC),2025. 4 4
[79] Yuhao Wu, Franziska Roesner, Tadayoshi Kohno, Ning Zhang, and 3 3
Umar Iqbal. IsolateGPT: An Execution Isolation Architecture for
LLM-Based Agentic Systems. In Network and Distributed System 2 2
Security(NDSS)Symposium,2025.
1 1
[80] JieruiXie,BartPietKnijnenburg,andHongxiaJin. Locationsharing
privacy preference: analysis and personalized recommendation. In 0% 20% 40% 60% 80% 100% 0% 20% 40% 60% 80% 100%
Proceedingsofthe19thinternationalconferenceonIntelligentUser
(f) AI tool trust (g) Privacy consciousness
Interfaces,pages189–198,2014.
[81] ShunyuYao,JeffreyZhao,DianYu,NanDu,IzhakShafran,Karthik
Figure 8: Group users according to their demographic in-
Narasimhan, and Yuan Cao. React: Synergizing reasoning and act-
ing in language models. In International Conference on Learning formation, which includes age group, education level, and
Representations(ICLR),2023. sex,aswellastheirself-reportedmetricssuchasAItoolfa-
[82] YifanYao,JinhaoDuan,KaidiXu,YuanfangCai,ZhiboSun,andYue miliarity,AItoolusagefrequency,AItooltrust,andprivacy
Zhang.Asurveyonlargelanguagemodel(llm)securityandprivacy: consciousnesslevel.Foragegroups,participantsaredivided
Thegood,thebad,andtheugly. High-ConfidenceComputing,page into below 25, 25-39, 40-55, and over 55. Regarding edu-
100211,2024.
cation level, participants are categorized into four groups:
[83] NicoleZhan,StefanSarkadi,andJoseSuch. Privacy-enhancedper- highschoolorless(HS),bachelor’s(BS),master’s(MS),and
sonalassistantsbasedondialoguesandcasesimilarity. InEuropean
PhD or other doctorate (PhD). For sex, F denotes female
ConferenceonArtificialIntelligence.IOSPress,2023.
and M denotes male. Self-reported metrics are measured on
[84] XiaoZhan,StefanSarkadi,NataliaCriado,andJoseSuch. Amodel
a four-level scale (1-4) where a higher number indicates a
forgoverninginformationsharinginsmartassistants.InProceedings
ofthe2022AAAI/ACMConferenceonAI,Ethics,andSociety,pages greater level; for example, a privacy consciousness level of
845–855,2022. 4 signifies the highest concern for privacy.
[85] YanjieZhao,XinyiHou,ShenaoWang,andHaoyuWang. Llmapp
storeanalysis:Avisionandroadmap.ACMTransactionsonSoftware
EngineeringandMethodology,2024.
distributions across different levels of these metrics. As
showninFigure8d,participantsmorefamiliarwithAItools
Appendix A.
were more likely to select Always Allow and less likely to
Analysis of Demographics and Self-Reported chooseNeverShare.AlthoughAIusagefrequencyandtrust
Metrics (Figures 8e and 8f) do not show strong effects on overall
permissionpreferences,thosewhousedAItoolsmoreoften
We analyze each participant’s basic demographic infor- and reported higher trust levels tended to favor Always
mation and self-reported metrics related to AI usage and Allow. This suggests that greater engagement with AI may
privacy consciousness to investigate whether these factors correlate with a preference for smoother user experiences,
significantlyinfluencetheirpermissionpreferences(seeFig- enabled by more permissive data-sharing settings.
ure 8). Figure 8a shows that participants under the age of For the privacy consciousness (Figure 8g), participants
25 selected Always Allow permissions more frequently than withhigherlevelsofprivacyconsciousnessweremorelikely
those over 55 (35.0% vs. 25.5%), suggesting a generational to choose Never Share, reflecting a stronger desire to re-
difference in privacy attitudes. Additionally, participants strict data sharing. Interestingly, participants with medium
with higher levels of education granted fewer data-sharing levels of privacy consciousness selected Always Allow less
permissions overall (Figure 8b). often than those with high levels, but overall, they ex-
The self-reported metrics include AI tool familiarity, hibited a more permissive data-sharing pattern compared
usage frequency, trust in AI tools, and privacy conscious- to highly privacy-conscious individuals. While these self-
ness.Weobserveasignificantshiftinpermissionpreference reported measures may lack standardization or objectiv-
17

--- 第 18 页 ---

 
Health & Fitness 1.0
Work & Productivity
Finance
Shopping
Travel 0.8
Entertainment
Social e
WorHke &aS lPtmhro a&dr utF ciHttnoivemisteys   c valu0.6 hreshold
EnterStahFioinnTpmrapaneivncnegetl Metri0.4 APRF1rece ccscuacirslolairoceny Baseline t
Social
Smart Home FPR
  0.2 FNR
Health & Fitness Neg. region
Work & Productivity
Finance Pos. region
Shopping 0.0
Travel 0 1 2 3 4 5
Entertainment Threshold
Social
Smart Home Figure 10: Collaborative filtering model metrics under dif-
Figure 9: Heatmap of permission allowance rates across ferent prediction score thresholds.
various domains for individual users. The figure is seg-
mentedintothreedistinctusergroups.Ineachsegmentation
(user group), each column corresponds to a user and each threshold for each fold in five-fold cross-validation by se-
row to a distinct domain (Health & Fitness, Work & Pro- lecting the point where the FPR equals the FNR. Figure 10
ductivity, Finance, Shopping, Travel, Entertainment, Social, illustrates the CF model’s metrics under varying thresholds
Smart Home). Darker green indicates a higher permission for one of our test runs and shows the selected baseline
allowance rate, while darker red indicates a lower rate. threshold for reporting the model comparison results. For
integrating the IC with CF, we use only CF predictions
with high confidence. In the figure, the negative region and
ity [14], and some participants may have inaccurate percep- positive region define ranges of prediction scores that serve
tions of their privacy attitudes, the overall trends provide as additional contextual examples for the hybrid model.
useful insights into potential user data-sharing behaviors. In
summary, our analysis indicates that demographic factors,
AI-related attitudes, and privacy consciousness are associ-
atedwithparticipants’data-sharingpreferences.Thesefind-
ings help us identify patterns that can be used to anticipate
general user attitudes toward data-sharing permissions.
Appendix B.
Analysis of User Permission Preferences Across
Domains
Figure 9 provides insights into data-sharing permission
preferences across domains among participants. It reveals a
high degree of consistency in overall permission behaviors,
as many participants exhibit uniform preferences across
different domains, evident from extensive areas of homo-
geneous coloring. At the same time, variations in color
intensity highlight specific domains, such as Finance and
Shopping, where permission preferences significantly differ
from more permissive areas like Health & Fitness and
Social for some participants. These patterns emphasize that
while participants generally maintain consistent permission
preferences across domains, the distinct contexts of each
domaincanleadtoobservablevariationsinthedata-sharing
preferences of some participants.
Appendix C.
CF Model Prediction Threshold
Since the CF model’s performance is highly sensitive
to the prediction score threshold, we establish a baseline
18

--- 第 19 页 ---

Appendix D.
Meta-Review
Thefollowingmeta-reviewwaspreparedbytheprogram
committee for the 2026 IEEE Symposium on Security and
Privacy (S&P) as part of the review process as detailed in
the call for papers.
D.1. Summary
This paper presents an automated permission man-
agement system for AI agents. The authors conducted a
vignette-based survey of 205 users across a range of do-
mains and scenarios to understand how users make permis-
sion decisions regarding data access and sharing in agentic
systems. Building on these findings, the paper presents a
preliminary design of a permission prediction model.
D.2. Scientific Contributions
• Independent Confirmation of Important Results with
Limited Prior Research
• Establishes a New Research Direction
D.3. Reasons for Acceptance
1) The paper builds on a long line of work that seeks to
understandfactorsinfluencingusers’decisionsinterms
of sharing data and granting permissions to automated
systems. Prior work in this space has focused on non-
LLM-based personal assistants, and this paper obtains
results for users’ preferences in LLM-based agentic
systems.
2) Thispaperisoneofthefirsttoinvestigatetheproblem
ofautomatingpermissionmanagementforAIagents.It
arguesthat,sinceautomationisakeyvalueproposition
of AI agents, there is a need for permission manage-
mentsystemsthatcanautomaticallymakedecisionson
user’s behalf. The paper establishes this new subfield
of automated permission systems for AI agents, and
presents one point in the design space of this problem.
19