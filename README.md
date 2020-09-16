# Analogical-Rule-Mining-on-Knowledge-Graphs
Enrich KB's with /similar_to links for better rule mining

This is the first attempt at Analogical Reasoning (AR) on Knowledge Graphs (KG).
Here we experimented with enriching the KG with *similar_to* links b/w entities and then mine rules with out-of-the box rule mining systems (e.g AMIE+). To quantify similarity b/w entities, we employed graph embedding models (e.g. DistMult) or natural language models(e.g. word2vec) to obtain vector representations for the graph and then used similarity based functions to generate score for entity pairs. We also experimented with a discrete notion of similarity as our last method which seeks to capture similarity b/w pairs while paying attention to their role (i.e. subject vs object).

The adopted datasets for our experiments were FB15K, FB15K237, and WN18RR. Each dataset was enriched with progressive levels of similarity links to understand the effect of enrichment levels and avoid saturation. These enrichment levels, denoted by $\sigma$ were chosen as a percentage of the size of the test sets for each benchmark (recent experiments used $\sigma =$ %5, %10, %15, %20, %50, and %100).

Models were evaluated in a link prediction task, with hit@10 scores as the measure for comparison with baseline models. Overall, the gains in hit@10, although mostly positive, but were lower than expected and led us to further investigation of the concept of AR on knowledge graphs.


-----

The most recent and complete version of our codes and experiments is kept on the *Colab_PyTorch* branch of the repo. Entire functionality of our module and every piece of the pipeline is implemented in a class named xClass under pkgX folder. Instances of X allow installation of required modules to the enrichment and testing of generated rules. Jupyter notebooks provide some examples for using X objects.


