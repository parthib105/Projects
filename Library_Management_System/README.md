# Library Management System

## Description
This is a simple Library Management System implemented in C++. It allows users to manage books and user data in a library, including operations for adding books, adding users, borrowing books, returning books, and searching for books.

## Table of Contents
- [Classes and Attributes](#classes-and-attributes)
  - [Book Class](#book-class)
  - [User Class](#user-class)
  - [Library Class](#library-class)
- [Operations](#operations)
  - [Adding a Book](#adding-a-book)
  - [Adding a User](#adding-a-user)
  - [Borrowing a Book](#borrowing-a-book)
  - [Returning a Book](#returning-a-book)
  - [Searching for a Book](#searching-for-a-book)

## Classes and Attributes

### Book Class
- **Attributes**:
  - `title`: Title of the book (string).
  - `author`: Author of the book (string).
  - `ISBN`: Unique identifier of the book (string).
  - `availability`: Number of copies available (integer).

### User Class
- **Attributes**:
  - `user_id`: Unique identifier of the user (string).
  - `name`: Name of the user (string).
  - `borrowed_books`: List of ISBNs of books borrowed by the user (vector of strings).

### Library Class
- **Attributes**:
  - `books`: HashMap to store books (`ISBN` -> `Book`).
  - `users`: HashMap to store users (`user_id` -> `User`).

## Operations

### Adding a Book
To add a new book to the library:
```cpp
library.add_book("The Catcher in the Rye", "J.D. Salinger", "9780316769488", 5);
