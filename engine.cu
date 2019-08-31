#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <random>

#define MAX_DENSITY (64)

#define MEAN_VELOCITY (0.02)
#define DAMPING (0.999)
#define DT (0.01)

typedef struct Particle
{
    float pos[DIMS];
    float vel[DIMS];
} Particle;

typedef struct Cell
{
    int count;
    Particle particles[MAX_DENSITY];
} Cell;


__device__
void collide(Particle *a, Particle *b) {
    // DIMS = 2
    float dx = a->pos[0] - b->pos[0];
    float dy = a->pos[1] - b->pos[1];

    float distance = sqrt(dx*dx + dy*dy);
    if (distance < RADIUS * GRID_WIDTH) {
        dx /= distance;
        dy /= distance;

        float vx = a->vel[0] - b->vel[0];
        float vy = a->vel[1] - b->vel[1];

        float u = (dx * vx + dy * vy) * DAMPING;
        float ux = dx * u;
        float uy = dy * u;

        a->vel[0] -= ux;
        a->vel[1] -= uy;
        b->vel[0] += ux;
        b->vel[1] += uy;

        float excess = RADIUS * GRID_WIDTH - distance;

        a->pos[0] += 0.5 * excess * dx;
        a->pos[1] += 0.5 * excess * dy;
        b->pos[0] -= 0.5 * excess * dx;
        b->pos[1] -= 0.5 * excess * dy;
    }
}

__global__
void step(Cell *cells) {
    // DIMS = 2
    int index = threadIdx.x + GRID_WIDTH * blockIdx.x;

    // Move particles
    for (int i = 0; i < cells[index].count; ++i) {
        for (int j = 0; j < DIMS; ++j) {
            cells[index].particles[i].pos[j] += DT * cells[index].particles[i].vel[j];
        }
    }

    // Figure out grid neighbours
    // Diagonals intentionally omitted in favor of efficiency at the cost of accuracy
    int west = index - 1;
    if (threadIdx.x == 0) {
        west = GRID_WIDTH - 1 + GRID_WIDTH * blockIdx.x;
    }
    int north = index - GRID_WIDTH;
    if (blockIdx.x == 0) {
        north = threadIdx.x + GRID_WIDTH * (GRID_WIDTH - 1);
    }

    // We only need pairwise collisions so we can omit the reverse pass over the grid.
    // int east = index + 1;
    // if (threadIdx.x == GRID_WIDTH - 1) {
    //     east = blockIdx.x * GRID_WIDTH;
    // }
    // int south = index + GRID_WIDTH;
    // if (blockIdx.x == GRID_WIDTH - 1) {
    //     south = threadIdx.x;
    // }

    for (int i = 0; i < cells[index].count; ++i) {
        for (int j = i + 1; j < cells[index].count; ++j) {
            collide(cells[index].particles + i, cells[index].particles + j);
        }
        for (int j = 0; j < cells[west].count; ++j) {
            collide(cells[index].particles + i, cells[west].particles + j);
        }
        for (int j = 0; j < cells[north].count; ++j) {
            collide(cells[index].particles + i, cells[north].particles + j);
        }
        // See comment about pairwise collisions above on why these can be omitted.
        // for (int j = 0; j < cells[east].count; ++j) {
        //     collide(cells[index].particles + i, cells[east].particles + j);
        // }
        // for (int j = 0; j < cells[south].count; ++j) {
        //     collide(cells[index].particles + i, cells[south].particles + j);
        // }
    }
}

void migrate(Cell *cells) {
    for (int i = 0; i < GRID_WIDTH; ++i) {
        for (int j = 0; j < GRID_WIDTH; ++j) {
            int index = i + GRID_WIDTH * j;
            int west = i - 1 + GRID_WIDTH * j;
            if (i == 0) {
                west = GRID_WIDTH - 1 + GRID_WIDTH * j;
            }
            int east = i + 1 + GRID_WIDTH * j;
            if (i == GRID_WIDTH - 1) {
                east = GRID_WIDTH * j;
            }
            int north = i + (j - 1) * GRID_WIDTH;
            if (j == 0) {
                north = i + GRID_WIDTH * (GRID_WIDTH - 1);
            }
            int south = i + (j + 1) * GRID_WIDTH;
            if (j == GRID_WIDTH - 1) {
                south = i;
            }

            // There are no diagonal migrations. It is hoped that those will resolve themselves in two steps.
            // assert(cells[index].count >= 0);
            // assert(cells[index].count <= MAX_DENSITY);
            for (int k = 0; k < cells[index].count; ++k) {
                int drop = 0;
                if (cells[index].particles[k].pos[0] < i) {
                    if (cells[west].count == MAX_DENSITY) {
                        continue;
                    }
                    if (cells[index].particles[k].pos[0] < 0) {
                        cells[index].particles[k].pos[0] += GRID_WIDTH;
                    }
                    cells[west].particles[cells[west].count] = cells[index].particles[k];
                    cells[west].count += 1;
                    drop = 1;
                } else if (cells[index].particles[k].pos[0] >= i + 1) {
                    if (cells[east].count == MAX_DENSITY) {
                        continue;
                    }
                    if (cells[index].particles[k].pos[0] >= GRID_WIDTH) {
                        cells[index].particles[k].pos[0] -= GRID_WIDTH;
                    }
                    cells[east].particles[cells[east].count] = cells[index].particles[k];
                    cells[east].count += 1;
                    drop = 1;
                } else if (cells[index].particles[k].pos[1] < j) {
                    if (cells[north].count == MAX_DENSITY) {
                        continue;
                    }
                    if (cells[index].particles[k].pos[1] < 0) {
                        cells[index].particles[k].pos[1] += GRID_WIDTH;
                    }
                    cells[north].particles[cells[north].count] = cells[index].particles[k];
                    cells[north].count += 1;
                    drop = 1;
                } else if (cells[index].particles[k].pos[1] >= j + 1) {
                    if (cells[south].count == MAX_DENSITY) {
                        continue;
                    }
                    if (cells[index].particles[k].pos[1] >= GRID_WIDTH) {
                        cells[index].particles[k].pos[1] -= GRID_WIDTH;
                    }
                    cells[south].particles[cells[south].count] = cells[index].particles[k];
                    cells[south].count += 1;
                    drop = 1;
                }

                if (drop) {
                    cells[index].count -= 1;
                    for (int l = k; l < cells[index].count; ++l) {
                        cells[index].particles[l] = cells[index].particles[l + 1];
                    }
                }
            }
        }
    }
}

void manipulate(Cell *cells)
{
    for (int index = 0; index < GRID_WIDTH * GRID_WIDTH; ++index) {
        for (int k = 0; k < cells[index].count; ++k) {
            Particle p = cells[index].particles[k];
            if (p.pos[0] < GRID_WIDTH * 0.2 && p.pos[1] > GRID_WIDTH * 0.45 && p.pos[1] < GRID_WIDTH * 0.55) {
                cells[index].particles[k].vel[0] += DT * GRID_WIDTH * 0.03;
            }

            if (p.pos[0] < GRID_WIDTH * 0.01 && p.vel[0] < 0) {
                cells[index].particles[k].vel[0] = -p.vel[0];
            }
            if (p.pos[0] > GRID_WIDTH - GRID_WIDTH * 0.01 && p.vel[0] > 0) {
                cells[index].particles[k].vel[0] = -p.vel[0];
            }
            if (p.pos[1] < GRID_WIDTH * 0.01 && p.vel[1] < 0) {
                cells[index].particles[k].vel[1] = -p.vel[1];
            }
            if (p.pos[1] > GRID_WIDTH - GRID_WIDTH * 0.01 && p.vel[1] > 0) {
                cells[index].particles[k].vel[1] = -p.vel[1];
            }
        }
    }
}

int main(void)
{
    Cell *cells;

    std::random_device dev;
    std::mt19937 rng(dev());
    std::normal_distribution<float> ndist(0, 1);
    std::uniform_real_distribution<float> udist(0, 1);

    // DIMS = 2
    cudaMallocManaged(&cells, GRID_WIDTH * GRID_WIDTH);

    for (int i = 0; i < GRID_WIDTH; ++i) {
        for (int j = 0; j < GRID_WIDTH; ++j) {
            int index = i + GRID_WIDTH * j;
            cells[index].count = DENSITY;
            for (int k = 0; k < cells[index].count; ++k) {
                cells[index].particles[k].pos[0] = i + udist(rng);
                cells[index].particles[k].pos[1] = j + udist(rng);
                cells[index].particles[k].vel[0] = ndist(rng) * MEAN_VELOCITY * GRID_WIDTH;
                cells[index].particles[k].vel[1] = ndist(rng) * MEAN_VELOCITY * GRID_WIDTH;
            }
        }
    }


    for (int j = 0; j < ROUNDS; ++j) {
        cudaDeviceSynchronize();
        step<<<GRID_WIDTH, GRID_WIDTH>>>(cells);
        cudaDeviceSynchronize();
        manipulate(cells);
        migrate(cells);

        int total_count = 0;
        for (int index = 0; index < GRID_WIDTH * GRID_WIDTH; ++index) {
            total_count += cells[index].count;
            for (int k = 0; k < cells[index].count; ++k) {
                fwrite(cells[index].particles[k].pos, sizeof(float), DIMS, stdout);
            }
        }
        assert(total_count == GRID_WIDTH*GRID_WIDTH * DENSITY);
    }

    cudaFree(cells);

    return 0;
}
