# AI-based Wind Turbine Question-Answering System
This repository contains supplementary resources for our paper "Automated Question-Answering for Interactive Decision Support in Operations & Maintenance of Wind Turbines" (in submission)

Contents of the repository include:-

(a) Original templates of natural language questions-Cypher query pairs.

(b) Templates after replacing with the vital metrics (e.g. alarm names, SCADA descriptions etc.)

(c) Paraphrased version of the dataset (augmented).


More details on the knowledge graph used for information retrieval based on user-provided queries can be found at https://github.com/joyjitchatterjee/XAI4Wind. (Chatterjee, J. and Dethlefs, N., “XAI4Wind: A Multimodal Knowledge Graph Database for Explainable Decision Support in Operations & Maintenance of Wind Turbines”, arXiv e-prints, 2020.)

# Acknowledgments
We acknowledge the publicly available Skillwind maintenance manual [1] and ORE Catapult's Platform for Operational Data (POD) [2] for the valuable resources used in development of this repository.
A subset of 102 SCADA features (with their names publicly available on POD) was used in developing the query templates. The maintenance actions segment is used to organise the information present in the Skillwind manual into a domain-specific ontology. Due to confidentiality reasons, we have not provided the numeric values and complete SCADA corpus, but only released information which is presently in the public domain in this repository. More information can be found in references (1) for the maintenance action manual and (2) for multiple other potential SCADA features and alarms available in POD.

## Note
Some information on SCADA features and alarms have been anonymised in the provided dataset. More information on these are available on the Platform for Operational Data [2]. The generated paraphrased versions can have occasional inconsistencies as would be expected with inference from a large language model in our domain-specific context. In some occasions, it may be possible that the user asks a question for which there is no relevant information available in the KG database. We have particularly observed inconsistencies in learning to generate numbers (e.g. alarm codes) for paraphrased versions of original templates, although many of these were minor (<5% cases) in the total dataset. Regardless, in this situation, the appropriate Cypher queries would be generated by the model based on the context of the original templates which the AI models learn from.

## References
1. Skillwind Manual (https://skillwind.com/wp-content/uploads/2017/08/SKILWIND_Maintenance_1.0.pdf) 
2. Platform for Operational Data (https://pod.ore.catapult.org.uk). 

# License

This repo is based on the MIT License, which allows free use of the provided resources, subject to the original sources being credit/acknowledged appropriately. The software/resources under MIT license is provided as is, without any liability or warranty at the end of the authors.
