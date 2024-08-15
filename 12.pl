% Define the goal condition
goal(5). % The goal node is 3

% Define the successor relationships
s(1, 2).
s(2, 3).
s(1, 4).
s(4, 5).
s(5, 3).
s(4, 6).

% solve(Node, Solution):
%   Solution is an acyclic path (in reverse order) between Node and a goal
solve(Node, Solution) :-
    depthfirst([], Node, Solution).

% depthfirst(Path, Node, Solution):
%   Extending the path [Node | Path] to a goal gives Solution
depthfirst(Path, Node, [Node | Path]) :-
    goal(Node).

depthfirst(Path, Node, Sol) :-
    s(Node, Node1),
    \+ member(Node1, Path),                % Prevent a cycle
    depthfirst([Node | Path], Node1, Sol).

% depthfirst2(Node, Solution, Maxdepth):
%   Solution is an acyclic path (in reverse order) between Node and a goal within Maxdepth
depthfirst2(Node, [Node], _) :-
    goal(Node).

depthfirst2(Node, [Node | Sol], Maxdepth) :-
    Maxdepth > 0,
    s(Node, Node1),
    Max1 is Maxdepth - 1,
    depthfirst2(Node1, Sol, Max1).

% Example Queries:

% To find a solution path from node 1 to the goal:
% ?- solve(1, Solution).

% To find a solution path from node 1 to the goal with a maximum depth of 3:
% ?- depthfirst2(1, Solution, 3).
