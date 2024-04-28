# Data Science Cheatsheet

## Background

I am currently doing a data science course. That's why I'm creating a series of cheat sheets with content that will help me and perhaps others to learn or simply for quick reference. 

## Cheat Sheets

* Python for Data Science
  * numpy
  * pandas
    * Datframes
    * Series
  * sklearn
  * seaborn
  * statsmodels.api
  * scipy.stats
* Theory on Statistical Tests
* Theory on Data Quality

## How to create an HTML or PDF file

> **Caution**
>
> I havn't found a simple way to create HTML where the latex formulas are converted correctly. By simple I mean using a script.
> 
> For good result I recommend to use the VS Code plugin "Markdown All in One".
> Feel free to contribute a better solution for a solution on a scripting basis.

### Preconditions

Install the following tools:

* https://wkhtmltopdf.org/downloads.html
* https://github.com/sgaunet/mdtohtml
  ```
  go install github.com/sgaunet/mdtohtml@latest
  ```

### Using the prepared scripts

To create an HTML version of the markdown file:

```
scripts/mkhtml

# or if you like to have table of contents
scripts/mkhtml --toc
```

To create a PDF version of the markdown file:

```
scripts/mkpdf
```
