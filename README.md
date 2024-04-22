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

## How to create an HTML or PDF file

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
```

To create a PDF version of the markdown file:

```
scripts/mkpdf
```
