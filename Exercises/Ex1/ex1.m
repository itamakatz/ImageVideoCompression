\documentclass[10pt]{article}
\usepackage[a4paper]{geometry}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{dirtytalk}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{hyperref}
\usepackage{float}

\setlength{\oddsidemargin}{0in}
\setlength{\evensidemargin}{0in}
\setlength{\textheight}{9in}
\setlength{\textwidth}{6.5in}
\setlength{\topmargin}{-0.5in}

\newcommand{\startQ}[0]{Write a function with:}

\newcounter{question}
\newenvironment{QuestionEnv}[1][]{\setlength{\parindent}{0pt}\refstepcounter{question}\par\medskip
   \textbf{Q\thequestion. #1}\rmfamily}{\medskip}

\newcommand{\Q}[1]{{\begin{QuestionEnv}#1\end{QuestionEnv}}}

\title{Image and Video Compression}
\author{Ex1}
\date{March 2021}

\begin{document}

\maketitle

This is a simple exercise 
The aim of this exercise is to remind students 
In this exercise you will learn to use Matlab 

\section{Basic image manipulations}
\Q{Loading a BMP image in B\textbackslash W}
\begin{quote} 
    \startQ 
    \begin{itemize}
        \item Input: string representing a path to a BMP image.
        \item Output: Matrix.
        \item Functionality: Returns a B\textbackslash W representation of the image.
    \end{itemize} 
\end{quote}

\Q{Flip matrix} 
\begin{quote}
    \thequestion.a: \startQ
    \begin{itemize}
        \item Input: Matrix.
        \item Output: Matrix.
        \item Functionality: Returns the matrix flipped over the horizontal axis.
    \end{itemize}
    \thequestion.b: \startQ
    \begin{itemize}
        \item Input: Matrix.
        \item Output: Matrix.
        \item Functionality: Returns the matrix flipped over the vertical axis.
    \end{itemize}    
\end{quote}

\Q{Negative image}
\begin{quote} 
    \startQ
    \begin{itemize}
        \item Input: Matrix representing an image.
        \item Output: Matrix.
        \item Functionality: Returns negative of the image.
    \end{itemize} 
    Tip: Recall the importance of the type of a matrix, and notice the max value of the matrix data type.
\end{quote}

\Q{White frame\textbackslash padding}
\begin{quote} 
    \startQ
    \begin{itemize}
        \item Input: Matrix of size (m,n), and an integer $d\geq0$
        \item Output: A \textbf{larger} matrix with size $(m+2d,n+2d)$.
        \item Functionality: Returns the matrix with additional \textbf{white} padding surrounding the image of length $d$ from all sides.
    \end{itemize} 
\end{quote}

\section{Using Structure Arrays}

\Q{Binning a matrix}
\begin{quote} 
    \startQ
    \begin{itemize}
        \item Input: Matrix of size (m,n), and $d>0$ s.t. $n~(\text{mod } d) = m~(\text{mod } d )= 0$ .
        \item Output: A struct array of size $(m/d,n/d)$.
        \item Functionality: The function splits the matrix into $d\times d$ matrices and save each of them as a struct.
    \end{itemize} 
    Tip: Since structure arrays contain data in fields that you access by name, you may want to \say{save} other info in additional fields for future use (see next question).
\end{quote}

\Q{Un-binning back to matrix}
\begin{quote} 
    \startQ
    \begin{itemize}
        \item Input: A struct array with the output of the last question
        \item Output: A matrix
        \item Functionality: Reconstructs a matrix out of the struct array.
    \end{itemize} 
\end{quote}

\section{Combining everything together}
In this section, you need to combine what you implemented in the last sections.
Tip: It might be useful to send a function as a parameter which is done in Matlab by \say{@foo} for some function foo. (For those who really want to make it as functional as possible, I recommend you look at how to use \href{http://www.latex-tutorial.com}{curry functions in Matlab}. )

\Q{Clear a cell}
\begin{quote} 
    \startQ
    \begin{itemize}
        \item Input: A BMP image, and $d>0$ s.t. $n~(\text{mod } d) = m~(\text{mod } d )= 0$, and indexes $i,j$
        \item Output: A BMP image of the same size
        \item Functionality: The functions simply colors the block $i,j$ with white
    \end{itemize}
    Example:
    \begin{figure}[H]
        \centering
        \begin{subfigure}{.5\textwidth}
          \centering
          \includegraphics[width=.6\textwidth]{Images/haze.png}
          \caption{Before}
        \end{subfigure}%
        \begin{subfigure}{.5\textwidth}
          \centering
          \includegraphics[width=.6\linewidth]{Images/ClearedCell.png}
          \caption{After}
        \end{subfigure}
        \caption{Before and after with $d=5$ and $\{i,j\}=\{3,3\}$}
    \end{figure}
\end{quote}


\Q{Flipping sub matrices}
\begin{quote} 
    \startQ
    \begin{itemize}
        \item Input: A BMP image, and $d>0$ s.t. $n~(\text{mod } d) = m~(\text{mod } d )= 0$ .
        \item Output: A BMP image of the same size
        \item Functionality: The functions splits the image, flips each sub-image in both axis directions, plots the new figure and returns the resulting image.
    \end{itemize} 
    Example:
    \begin{figure}[H]
        \centering
        \begin{subfigure}{.5\textwidth}
          \centering
          \includegraphics[width=.6\textwidth]{Images/haze.png}
          \caption{Before}
        \end{subfigure}%
        \begin{subfigure}{.5\textwidth}
          \centering
          \includegraphics[width=.6\linewidth]{Images/Flipped.png}
          \caption{After}
        \end{subfigure}
        \caption{Before and after with $d=5$}
    \end{figure}
\end{quote}

\Q{Padding sub matrices}
\begin{quote} 
    \startQ
    \begin{itemize}
        \item Input: A BMP image, and $d>0$ s.t. $n~(\text{mod } d) = m~(\text{mod } d )= 0$, and $c>0$.
        \item Output: A \textbf{larger} BMP image
        \item Functionality: The functions splits the image, pads from all directions of length $c$ each sub matrix with white, plots the new figure and returns the resulting image.
    \end{itemize} 
    Example:
    \begin{figure}[H]
        \centering
        \begin{subfigure}{.5\textwidth}
          \centering
          \includegraphics[width=.6\textwidth]{Images/haze.png}
          \caption{Before}
        \end{subfigure}%
        \begin{subfigure}{.5\textwidth}
          \centering
          \includegraphics[width=.6\linewidth]{Images/Padded.png}
          \caption{After}
        \end{subfigure}
        \caption{Before and after with $d=5$ and $c=1$}
    \end{figure}
\end{quote}

\end{document}