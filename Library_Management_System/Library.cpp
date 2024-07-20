#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>

using namespace std;

// Class for Book
class Book 
{
public:
    string title;       // Title of the book (string)       
    string author;      // Author of the book (string)
    string ISBN;        // Unique identifier of the book (string)
    int availability;   // Number of copies available (integer)

    // Default constructor
    Book() : title(""), author(""), ISBN(""), availability(0) {}

    // Book class constructor
    Book(string title, string author, string ISBN, int availability)
    {
        this->title = title;
        this->author = author;
        this->ISBN = ISBN;
        this->availability = availability;
    }
};

// Class for User
class User 
{
public:
    string user_id;                     // Unique identifier of the user (string)
    string name;                        // Name of the user (string)
    vector<string> borrowed_books;      // List of ISBNs of books borrowed by the user (vector of strings)

    // Default constructor
    User() : user_id(""), name("") {}

    // User class constructor
    User(string user_id, string name)
    {
        this->user_id = user_id;
        this->name = name;
    }
};

// Class for Library Management System
class Library 
{
private:
    unordered_map<string, Book> books;      // HashMap to store books ('ISBN' -> 'Book').
    unordered_map<string, User> users;      // HashMap to store users ('user_id' -> 'User').

public:
    // Add a new book to the library
    void add_book(string title, string author, string ISBN, int availability)
    {
        Book new_book(title, author, ISBN, availability);
        books[ISBN] = new_book;
    }

    // Add a new user to the library
    void add_user(string user_id, string name)
    {
        User new_user(user_id, name);
        users[user_id] = new_user;
    }

    // Borrow a book from the library
    void borrow_book(string user_id, string ISBN)
    {
        if (books.find(ISBN) != books.end() && books[ISBN].availability > 0)
        {
            users[user_id].borrowed_books.push_back(ISBN);
            books[ISBN].availability--;
            cout << "Book id : " << ISBN << ", is borrowed successfully." << endl;
            return;
        }
        
        cout << "Book id : " << ISBN << ", is not available for borrowing." << endl;
    }

    // Return a borrowed book to the library
    void return_book(string user_id, string ISBN)
    {
        vector<string>::iterator it = find(users[user_id].borrowed_books.begin(), users[user_id].borrowed_books.end(), ISBN);
        if (it != users[user_id].borrowed_books.end())
        {
            users[user_id].borrowed_books.erase(it);
            books[ISBN].availability++;
            cout << "Book id : " << ISBN << ", is returned successfully." << endl;
            return;
        }
            
        cout << "Book id : " << ISBN << ", is not borrowed by this user." << endl;
    }

    // Search for a book in the library by ISBN
    Book* search_book(string ISBN)
    {
        if (books.find(ISBN) != books.end())
        {
            return &books[ISBN];
        }

        cout << "Book id : " << ISBN << ", is not found." << endl;
        return NULL;
    }
};
