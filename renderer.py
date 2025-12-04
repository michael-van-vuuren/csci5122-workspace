import pygame
import math

# the renderer uses pygame to draw the moving and static objects
class Renderer:
    def __init__(self, screen):
        self.screen = screen
        self.info_font = pygame.font.SysFont(None, 24)
        self.big_font = pygame.font.Font(None, 74)
        self.font = self.info_font
        
    def draw(self, objects):
        shapes = []
    
        # background, ground, platform, and cow
        shapes.append(objects.background.get_shapes())
        shapes.append(objects.ground.get_shapes())
        shapes.append(objects.platform.get_shapes())
        shapes.extend(objects.cow.get_shapes())
        
        # clouds 
        for cloud in objects.clouds:
            shapes.extend(cloud.get_shapes())
        
        # rocket (with flame and crash message)
        rocket_shapes = objects.rocket.get_shapes()
        for shape in rocket_shapes:
            if shape['type'] == 'text':
                self.font = self.big_font
        shapes.extend(rocket_shapes)

        for shape in shapes:
            self.draw_shape(shape)

        # top left info
        self.font = self.info_font
        info_shapes = objects.info.get_shapes(objects.rocket)

        for shape in info_shapes:
            self.draw_shape(shape)

        pygame.display.flip()

    def draw_shape(self, shape):
        shape_type = shape['type']
        
        # background
        if shape_type == 'fill':
            self.screen.fill(shape['color'])

        # rectangles
        elif shape_type == 'rect':
            pygame.draw.rect(self.screen, shape['color'], shape['rect'])

        # ovals
        elif shape_type == 'ellipse':
            pygame.draw.ellipse(self.screen, shape['color'], shape['rect'])

        # circles
        elif shape_type == 'circle':
            if 'alpha' in shape:
                r = shape['radius']
                surface = pygame.Surface((r*2, r*2), pygame.SRCALPHA)
                pygame.draw.circle(surface, shape['color'], (r, r), r)
                self.screen.blit(surface, (shape['position'][0], shape['position'][1]))
            else:
                pygame.draw.circle(self.screen, shape['color'], shape['position'], shape['radius'], shape.get('width', 0))

        # straight lines
        elif shape_type == 'line':
            pygame.draw.line(self.screen, shape['color'], shape['start'], shape['end'], shape['width'])

        # flame triangles
        elif shape_type == 'polygon':
            pygame.draw.polygon(self.screen, shape['color'], shape['points'])

        # the rocket body
        elif shape_type == 'rocket':
            surf = pygame.Surface(shape['size'], pygame.SRCALPHA)
            pygame.draw.rect(surf, shape['color'], (0, 0, shape['size'][0], shape['size'][1]))
            rotated = pygame.transform.rotate(surf, -math.degrees(shape['angle']))
            rect = rotated.get_rect(center=shape['center'])
            self.screen.blit(rotated, rect)

        # text
        elif shape_type == 'text':
            shadow = self.font.render(shape['content'], True, (0, 0, 0))
            text = self.font.render(shape['content'], True, shape['color'])
            if 'center' in shape:
                s_rect = shadow.get_rect(center=shape['center'])
                t_rect = text.get_rect(center=shape['center'])
            else:
                s_rect = pygame.Rect(shape['position'], (0,0))
                t_rect = pygame.Rect(shape['position'], (0,0))
            offset = shape.get('shadow_offset', (2, 2))
            self.screen.blit(shadow, (s_rect.x + offset[0], s_rect.y + offset[1]))
            self.screen.blit(text, t_rect)
