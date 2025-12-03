import pygame
import random

def draw_cow(screen, ground_y):
    cow_x = 480
    cow_y = ground_y - 50
    BODY = (255, 255, 255)
    SPOT = (33, 33, 33)
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    PINK = (255, 180, 200)

    body_rect = pygame.Rect(cow_x, cow_y, 80, 40)
    pygame.draw.ellipse(screen, BODY, body_rect)
    pygame.draw.circle(screen, SPOT, (cow_x + 20, cow_y + 20), 10)
    pygame.draw.circle(screen, SPOT, (cow_x + 50, cow_y + 15), 12)

    head_rect = pygame.Rect(cow_x - 30, cow_y + 5, 35, 28)
    pygame.draw.ellipse(screen, BODY, head_rect)
    snout_rect = pygame.Rect(cow_x - 30, cow_y + 18, 30, 18)
    pygame.draw.ellipse(screen, PINK, snout_rect)

    ear_rect1 = pygame.Rect(cow_x - 22, cow_y - 3, 6, 9)
    ear_rect2 = pygame.Rect(cow_x - 8, cow_y - 3, 6, 9)
    pygame.draw.ellipse(screen, BLACK, ear_rect1)
    pygame.draw.ellipse(screen, BLACK, ear_rect2)

    ear_rect1 = pygame.Rect(cow_x - 28, cow_y + 2, 10, 6)
    ear_rect2 = pygame.Rect(cow_x - 6, cow_y + 2, 10, 6)
    pygame.draw.ellipse(screen, WHITE, ear_rect1)
    pygame.draw.ellipse(screen, WHITE, ear_rect2)

    eye_rect1 = pygame.Rect(cow_x - 22, cow_y + 10, 6, 10)
    eye_rect2 = pygame.Rect(cow_x - 10, cow_y + 10, 6, 10)
    pygame.draw.ellipse(screen, BLACK, eye_rect1)
    pygame.draw.ellipse(screen, BLACK, eye_rect2)

    pygame.draw.circle(screen, WHITE, (cow_x - 20, cow_y + 13), 1)
    pygame.draw.circle(screen, WHITE, (cow_x - 8,  cow_y + 13), 1)

    pygame.draw.circle(screen, SPOT, (cow_x - 17, cow_y + 25), 2)
    pygame.draw.circle(screen, SPOT, (cow_x - 11,  cow_y + 25), 2)

    for lx in [16, 24, 58, 66]:
        pygame.draw.rect(screen, BODY, (cow_x + lx, cow_y + 30, 6, 20))
        pygame.draw.rect(screen, BLACK, (cow_x + lx, cow_y + 50, 6, 5))

    pygame.draw.line(screen, BLACK, (cow_x + 75, cow_y + 10), (cow_x + 85, cow_y + 25), 3)
    tail_rect = pygame.Rect(cow_x + 85, cow_y + 25, 8, 4)
    pygame.draw.ellipse(screen, BLACK, tail_rect)


# cloud class
class Cloud:
    def __init__(self, x, y, screen_w):
        self.x = x
        self.y = y
        self.screen_w = screen_w
        self.wind_offset = random.uniform(-3, 3)
        self.circles = self.generate_circles()

    def reset(self, left_side=True):
        if left_side:
            self.x = random.randint(-200, -100)
        else:
            self.x = random.randint(self.screen_w + 100, self.screen_w + 200)

        self.y = random.randint(10, 220)
        self.wind_offset = random.uniform(-3, 3)
        self.circles = self.generate_circles()

    def generate_circles(self):
        circles = []
        grid_w = random.randint(4, 5)
        grid_h = random.randint(2, 3)
        cell_w = 15
        cell_h = 8
        base_radius = random.randint(14, 20)
        pos_jitter = 4
        bottom_extra_jitter = 3
        radius_jitter = 2

        for gy in range(grid_h):
            for gx in range(grid_w):
                cx = gx * cell_w + random.randint(-pos_jitter, pos_jitter)
                cy = gy * cell_h + random.randint(-pos_jitter, pos_jitter)

                if gy == grid_h - 1:
                    spread_factor = 2.0
                    center_adjust = (grid_w - 1) * cell_w / 2
                    cx = (cx - center_adjust) * spread_factor + center_adjust
                    cy += 10 + random.randint(-bottom_extra_jitter, bottom_extra_jitter)
                    radius = int(base_radius * 1.5)
                else:
                    radius = base_radius

                radius += random.randint(-radius_jitter, radius_jitter)
                alpha = 225

                circles.append((cx, cy, radius, alpha))

        return circles

    def update(self, dt, wind_speed):
        self.x += (wind_speed + self.wind_offset) * dt * 3
        # wind going right/left
        if wind_speed >= 0:
            if self.x > self.screen_w + 150:
                self.reset(left_side=True)
        else:
            if self.x < -150:
                self.reset(left_side=False)

    def draw(self, screen):
        for ox, oy, r, a in self.circles:
            cloud_circle = pygame.Surface((r*2, r*2), pygame.SRCALPHA)
            pygame.draw.circle(cloud_circle, (255, 255, 255, a), (r, r), r)
            screen.blit(cloud_circle, (self.x + ox, self.y + oy))
