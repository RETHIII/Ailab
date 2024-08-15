:- use_module(library(clpfd)).

% Main predicate to solve the N-Queens problem
n_queens(N, Qs) :-
    length(Qs, N),              % Length of Qs should be N (number of queens)
    Qs ins 1..N,               % Queens should be within the board limits
    safe_queens(Qs),           % Ensure no two queens attack each other
    labeling([], Qs).          % Find solutions with labeling

% Base case: No queens means no constraints
safe_queens([]).

% Recursive case: Check constraints for each queen
safe_queens([Q|Qs]) :-
    safe_queens(Qs, Q, 1),     % Check each queen's position
    safe_queens(Qs).           % Continue for remaining queens

% Base case for checking constraints
safe_queens([], _, _).

% Recursive case for checking constraints between queens
safe_queens([Q|Qs], Q0, D0) :-
    Q0 #\= Q,                 % Ensure not in the same row
    abs(Q0 - Q) #\= D0,       % Ensure not on the same diagonal
    D1 #= D0 + 1,             % Increment diagonal distance
    safe_queens(Qs, Q0, D1).  % Continue checking for remaining queens
