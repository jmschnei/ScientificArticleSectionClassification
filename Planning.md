# Section Classification Planification

The further development of the Scientific Article Section Classification is defined and described using this document. 


## Goal of the approach

The main goal is to develop a tool that can classify sectiions of scientific publications. This classification has to be made using labels, which could be either predefined or dynamically updated. 


## Architecture

The functionality of the tool is depicted on the following figure:

FIGURE

As can be seen, the main functionality is divided into three modules:

## Offline Processing

- [ ] [Off-1.]() Label Generation using titles. The automatic generation of labesl throught he usage of existing datasets of scientific papers, where titles and texts of sections are already provided. The titles are extracted, ranked and filtered basedd on frequency. The mosot frequent ones are used as labels. 
  - Literature analysis about the topic to see where we are
  - 
  - 
- [ ] [Off-2.]() Manual Label Generation. Define manually a set of labels that correspoond to sections. 

## Dataset Annotation

- [ ] [DA-1.]() Data Selection. Which data can be used for an annotationo task? Alreddy processed papers? Papers in PDF format?
- [ ] [DA-2.]() Who is going to annotate the documents? Finding annotators.
- [ ] [DA-3.]() Preparation of the annotation environment (Inceptiono??).

## Model Training

- [ ] [MT-1.]() Model selection. 
- [ ] [MT-2.]() Data Preparation. 
- [ ] [MT-3.]() Training.

## Experiments

- [ ] [Exp-1.]()  
- [ ] [Exp-2.]()  
- [ ] [Exp-3.]() 

## Online Processing

- [ ] [On-1.]() Flask service. 
- [ ] [On-2.]() OpenAPI specification. 
- [ ] [On-3.]() Postman test collection.
- [ ] [On-4.]() Postman test collection.

## Paper

- [ ] [Pa-1.]() Create an overleaf document with a generic template (LNCS, ACL, etc.) 
- [ ] [Pa-2.]() Include related work.
- [ ] [Pa-3.]() 
- [ ] [Pa-4.]() 


## Multilinguality

- [ ] [Mu-1.]() 

## Roadmap

The plan for the development of the section classification tool is depicted in this section: 

TABLE including tasks, persons, etc.

| :-----: | :---: | :----: | :---: |
| Task  | Resposible Person | Start Date | Estimated Duration |
| :-----: | :---: | :----: | :---: |
| Mu-1. | Kate    | Mar 24 | 3W    |
| :-----: | :---: | :----: | :---: |


> [!NOTE]
> Useful information that users should know, even when skimming content.

> [!TIP]
> Helpful advice for doing things better or more easily.

> [!IMPORTANT]
> Key information users need to know to achieve their goal.

> [!WARNING]
> Urgent info that needs immediate user attention to avoid problems.

> [!CAUTION]
> Advises about risks or negative outcomes of certain actions.