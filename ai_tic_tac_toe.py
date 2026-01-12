import math
import sys
import pygame
import numpy as np
from typing import List, Tuple, Optional

# Initialize pygame
pygame.init()

# Constants
BOARD_SIZE = 3
WIDTH, HEIGHT = 600, 700
GRID_WIDTH = 6
CELL_SIZE = WIDTH // BOARD_SIZE
LINE_COLOR = (23, 145, 135)
BG_COLOR = (28, 40, 56)
PLAYER_COLOR = (84, 153, 199)  # Blue for X
AI_COLOR = (231, 76, 60)       # Red for O
TEXT_COLOR = (236, 240, 241)
HIGHLIGHT_COLOR = (46, 204, 113)

# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI Tic-Tac-Toe - Minimax Algorithm")
font = pygame.font.SysFont('Arial', 40)
small_font = pygame.font.SysFont('Arial', 30)

class TicTacToe:
    def __init__(self):
        self.board = [[' ' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.current_player = 'X'  # Human is X
        self.game_over = False
        self.winner = None
        self.last_ai_move = None
        self.moves_made = 0
        self.max_moves = BOARD_SIZE * BOARD_SIZE
        
        # Game statistics
        self.player_wins = 0
        self.ai_wins = 0
        self.draws = 0
        
        # Minimax statistics
        self.nodes_explored = 0
        self.depth_reached = 0
        
    def reset_game(self):
        self.board = [[' ' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.current_player = 'X'
        self.game_over = False
        self.winner = None
        self.last_ai_move = None
        self.moves_made = 0
        self.nodes_explored = 0
        self.depth_reached = 0
    
    def make_move(self, row: int, col: int, player: str) -> bool:
        if self.game_over or self.board[row][col] != ' ':
            return False
            
        self.board[row][col] = player
        self.moves_made += 1
        
        # Check for win
        if self.check_winner(player):
            self.game_over = True
            self.winner = player
            if player == 'X':
                self.player_wins += 1
            else:
                self.ai_wins += 1
        elif self.moves_made == self.max_moves:
            self.game_over = True
            self.draws += 1
            
        self.current_player = 'O' if player == 'X' else 'X'
        return True
    
    def check_winner(self, player: str) -> bool:
        # Check rows and columns
        for i in range(BOARD_SIZE):
            if all(self.board[i][j] == player for j in range(BOARD_SIZE)):
                return True
            if all(self.board[j][i] == player for j in range(BOARD_SIZE)):
                return True
        
        # Check diagonals
        if all(self.board[i][i] == player for i in range(BOARD_SIZE)):
            return True
        if all(self.board[i][BOARD_SIZE-1-i] == player for i in range(BOARD_SIZE)):
            return True
        
        return False
    
    def get_empty_cells(self) -> List[Tuple[int, int]]:
        return [(i, j) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE) 
                if self.board[i][j] == ' ']
    
    def minimax(self, depth: int, is_maximizing: bool, alpha: float = -math.inf, 
                beta: float = math.inf) -> Tuple[float, Optional[Tuple[int, int]]]:
        self.nodes_explored += 1
        self.depth_reached = max(self.depth_reached, depth)
        
        # Terminal states
        if self.check_winner('O'):  # AI wins
            return 10 - depth, None
        if self.check_winner('X'):  # Human wins
            return depth - 10, None
        if not self.get_empty_cells():  # Draw
            return 0, None
        
        best_move = None
        
        if is_maximizing:  # AI's turn (O)
            max_eval = -math.inf
            for row, col in self.get_empty_cells():
                self.board[row][col] = 'O'
                eval_score, _ = self.minimax(depth + 1, False, alpha, beta)
                self.board[row][col] = ' '
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = (row, col)
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Beta cut-off
                    
            return max_eval, best_move
        else:  # Minimizing (Human's turn - X)
            min_eval = math.inf
            for row, col in self.get_empty_cells():
                self.board[row][col] = 'X'
                eval_score, _ = self.minimax(depth + 1, True, alpha, beta)
                self.board[row][col] = ' '
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = (row, col)
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha cut-off
                    
            return min_eval, best_move
    
    def get_ai_move(self) -> Tuple[int, int]:
        """Get AI move using Minimax with Alpha-Beta pruning"""
        _, move = self.minimax(0, True)
        if move is None:  # Shouldn't happen in valid game state
            return self.get_empty_cells()[0]
        return move
    
    def is_valid_move(self, row: int, col: int) -> bool:
        return (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE 
                and self.board[row][col] == ' ')

def draw_board(game: TicTacToe):
    screen.fill(BG_COLOR)
    
    # Draw grid lines
    for i in range(1, BOARD_SIZE):
        # Vertical lines
        pygame.draw.line(screen, LINE_COLOR, 
                        (i * CELL_SIZE, 0), 
                        (i * CELL_SIZE, HEIGHT - 100), 
                        GRID_WIDTH)
        # Horizontal lines
        pygame.draw.line(screen, LINE_COLOR, 
                        (0, i * CELL_SIZE), 
                        (WIDTH, i * CELL_SIZE), 
                        GRID_WIDTH)
    
    # Draw X's and O's
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            cell_x = col * CELL_SIZE + CELL_SIZE // 2
            cell_y = row * CELL_SIZE + CELL_SIZE // 2
            
            if game.board[row][col] == 'X':
                # Draw X
                size = CELL_SIZE // 3
                pygame.draw.line(screen, PLAYER_COLOR,
                               (cell_x - size, cell_y - size),
                               (cell_x + size, cell_y + size), 8)
                pygame.draw.line(screen, PLAYER_COLOR,
                               (cell_x + size, cell_y - size),
                               (cell_x - size, cell_y + size), 8)
            elif game.board[row][col] == 'O':
                # Draw O
                radius = CELL_SIZE // 3
                pygame.draw.circle(screen, AI_COLOR, 
                                 (cell_x, cell_y), radius, 8)
    
    # Highlight last AI move
    if game.last_ai_move:
        row, col = game.last_ai_move
        rect_x = col * CELL_SIZE + 5
        rect_y = row * CELL_SIZE + 5
        pygame.draw.rect(screen, HIGHLIGHT_COLOR, 
                        (rect_x, rect_y, CELL_SIZE - 10, CELL_SIZE - 10), 3)
    
    # Draw game status and statistics
    status_y = HEIGHT - 90
    
    # Game status
    if game.game_over:
        if game.winner:
            status_text = f"{'You' if game.winner == 'X' else 'AI'} Wins!"
            status_color = PLAYER_COLOR if game.winner == 'X' else AI_COLOR
        else:
            status_text = "It's a Draw!"
            status_color = TEXT_COLOR
    else:
        status_text = f"Your Turn (X)" if game.current_player == 'X' else "AI Thinking..."
        status_color = PLAYER_COLOR if game.current_player == 'X' else AI_COLOR
    
    status_surface = font.render(status_text, True, status_color)
    screen.blit(status_surface, (WIDTH // 2 - status_surface.get_width() // 2, status_y))
    
    # Statistics
    stats_y = HEIGHT - 50
    stats_text = f"Wins: You {game.player_wins} - AI {game.ai_wins} | Draws: {game.draws}"
    stats_surface = small_font.render(stats_text, True, TEXT_COLOR)
    screen.blit(stats_surface, (WIDTH // 2 - stats_surface.get_width() // 2, stats_y))
    
    # Minimax info (only show when AI is playing)
    if game.current_player == 'O' and not game.game_over:
        info_y = 10
        info_text = f"Minimax Analysis: Nodes Explored: {game.nodes_explored}"
        info_surface = small_font.render(info_text, True, TEXT_COLOR)
        screen.blit(info_surface, (10, info_y))
        
        depth_text = f"Search Depth: {game.depth_reached}"
        depth_surface = small_font.render(depth_text, True, TEXT_COLOR)
        screen.blit(depth_surface, (10, info_y + 30))

def draw_button(text, x, y, width, height, color):
    pygame.draw.rect(screen, color, (x, y, width, height), border_radius=10)
    pygame.draw.rect(screen, TEXT_COLOR, (x, y, width, height), 2, border_radius=10)
    
    text_surface = small_font.render(text, True, TEXT_COLOR)
    text_rect = text_surface.get_rect(center=(x + width // 2, y + height // 2))
    screen.blit(text_surface, text_rect)
    
    return pygame.Rect(x, y, width, height)

def main():
    game = TicTacToe()
    clock = pygame.time.Clock()
    ai_thinking = False
    ai_think_time = 500 # milliseconds
    
    # Button rectangles
    reset_rect = pygame.Rect(WIDTH - 150, HEIGHT - 90, 140, 40)
    quit_rect = pygame.Rect(WIDTH - 150, HEIGHT - 40, 140, 40)
    
    # Main game loop
    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Check button clicks
                if reset_rect.collidepoint(mouse_pos):
                    game.reset_game()
                    ai_thinking = False
                elif quit_rect.collidepoint(mouse_pos):
                    running = False
                
                # Handle board clicks (only if human's turn and game not over)
                elif not game.game_over and game.current_player == 'X' and not ai_thinking:
                    col = mouse_pos[0] // CELL_SIZE
                    row = mouse_pos[1] // CELL_SIZE
                    
                    if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
                        if game.make_move(row, col, 'X'):
                            ai_thinking = True
                            pygame.time.set_timer(pygame.USEREVENT, ai_think_time)
            
            elif event.type == pygame.USEREVENT and ai_thinking:
                # AI's turn
                if not game.game_over and game.current_player == 'O':
                    ai_row, ai_col = game.get_ai_move()
                    game.make_move(ai_row, ai_col, 'O')
                    game.last_ai_move = (ai_row, ai_col)
                ai_thinking = False
                pygame.time.set_timer(pygame.USEREVENT, 0)  # Clear timer
        
        # Draw everything
        draw_board(game)
        
        # Draw buttons
        reset_color = (41, 128, 185) if reset_rect.collidepoint(mouse_pos) else (52, 152, 219)
        quit_color = (231, 76, 60) if quit_rect.collidepoint(mouse_pos) else (192, 57, 43)
        
        reset_rect = draw_button("New Game", WIDTH - 150, HEIGHT - 90, 140, 40, reset_color)
        quit_rect = draw_button("Quit", WIDTH - 150, HEIGHT - 40, 140, 40, quit_color)
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
