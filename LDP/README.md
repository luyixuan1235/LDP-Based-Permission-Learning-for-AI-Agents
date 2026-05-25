> This conference version of the paper has been accepted by IEEE ICDE 2025. 🎉

# Data Poisoning Attacks to Local Differential Privacy Protocols for Graphs
[[Technical Report](https://github.com/hahahumble/DPA2Graphs/blob/main/TechnicalReport.pdf)] [[Source Code](https://github.com/hahahumble/DPA2Graphs/tree/main/Code)]

## Abstract
Graph analysis has become increasingly popular with the prevalence of big data and machine learning. Traditional graph data analysis methods often assume the existence of a trusted third party to collect and store the graph data, which does not align with real-world situations. To address this, some research has proposed utilizing Local Differential Privacy (LDP) to collect graph data or graph metrics (e.g., clustering coefficient). This line of research focuses on collecting two atomic graph metrics (the adjacency bit vectors and node degrees) from each node locally under LDP to synthesize an entire graph or generate graph metrics. However, they have not considered the security issues of LDP for graphs.

In this paper, we bridge the gap by demonstrating that an attacker can inject fake users into LDP protocols for graphs and design data poisoning attacks to degrade the quality of graph metrics. In particular, we present three data poisoning attacks to LDP protocols for graphs. As a proof of concept, we focus on data poisoning attacks on two classical graph metrics: degree centrality and clustering coefficient. We further design two countermeasures for these data poisoning attacks. Experimental study on real-world datasets demonstrates that our attacks can largely degrade the quality of collected graph metrics, and the proposed countermeasures cannot effectively offset the effect, which calls for the development of new defenses.
## Requirement
- Java 17.0.1

## Usage
1. Clone this repository and navigate to `Code` directory.
```bash
git clone https://github.com/hahahumble/DPA2Graphs.git
cd DPA2Graphs/Code
```

2. Compile the Java source files.
```bash
javac -cp "lib/ujmp-core-0.3.0.jar" -d out src/com/graph/**/*.java
```

3. Run the program.
```bash
java -cp "out:lib/ujmp-core-0.3.0.jar" com.graph.Main.AttackDegreeCentrality
java -cp "out:lib/ujmp-core-0.3.0.jar" com.graph.Main.AttackLCC
java -cp "out:lib/ujmp-core-0.3.0.jar" com.graph.Main.PerturbedLCC
...
```