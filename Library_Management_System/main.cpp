#include "Library.cpp"
#include <fstream>
using namespace std;

int main()
{
    Library library;

    // Open input file
    ifstream input_file("input.txt");
    if (!input_file.is_open())
    {
        cerr << "Error opening input file." << endl;
        return 1;
    }

    string command;
    while (input_file >> command)
    {
        if (command == "add_book")
        {
            string title, author, ISBN;
            int availability;
            input_file >> title >> author >> ISBN >> availability;
            library.add_book(title, author, ISBN, availability);
        }
        else if (command == "add_user")
        {
            string user_id, name;
            input_file >> user_id >> name;
            library.add_user(user_id, name);
        }
        else if (command == "borrow_book")
        {
            string user_id, ISBN;
            input_file >> user_id >> ISBN;
            library.borrow_book(user_id, ISBN);
        }
        else if (command == "return_book")
        {
            string user_id, ISBN;
            input_file >> user_id >> ISBN;
            library.return_book(user_id, ISBN);
        }
        else
        {
            cerr << "Invalid command: " << command << endl;
        }
    }

    input_file.close();

    return 0;
}
