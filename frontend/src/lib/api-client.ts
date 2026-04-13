import { z } from 'zod';

export const VelocityPointSchema = z.object({
  x: z.number(),
  y: z.number(),
  z: z.number(),
  u: z.number(),
  v: z.number(),
  w: z.number(),
  p: z.number(),
  t: z.number(),
  magnitude: z.number()
});

export const SliceResponseSchema = z.object({
  data: z.array(VelocityPointSchema),
  grid_shape: z.tuple([z.number(), z.number()]),
  z_height: z.number(),
  domain_bounds: z.object({
    x_min: z.number(),
    x_max: z.number(),
    y_min: z.number(),
    y_max: z.number(),
    z_min: z.number(),
    z_max: z.number()
  }),
  statistics: z.object({
    mean_velocity: z.number(),
    max_velocity: z.number(),
    mean_pressure: z.number(),
    pressure_gradient: z.number()
  })
});

export type VelocityPoint = z.infer<typeof VelocityPointSchema>;
export type SliceResponse = z.infer<typeof SliceResponseSchema>;
