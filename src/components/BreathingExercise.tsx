"use client";

import { useState, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Wind, Play, Pause, RotateCcw, ChevronRight, Sparkles } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

const EXERCISES = [
  {
    name: "Box Breathing",
    description: "Inhale, hold, exhale, hold - equal 4s counts. Used by Navy SEALs to stay calm.",
    steps: [
      { action: "Inhale", duration: 4 },
      { action: "Hold", duration: 4 },
      { action: "Exhale", duration: 4 },
      { action: "Hold", duration: 4 },
    ],
  },
  {
    name: "4-7-8 Breathing",
    description: "Relaxing breath. Inhale for 4, hold for 7, exhale for 8.",
    steps: [
      { action: "Inhale", duration: 4 },
      { action: "Hold", duration: 7 },
      { action: "Exhale", duration: 8 },
    ],
  },
  {
    name: "Calm Breath",
    description: "Simple rhythmic breathing for general relaxation.",
    steps: [
      { action: "Inhale", duration: 4 },
      { action: "Exhale", duration: 6 },
    ],
  },
];

export function BreathingExercise() {
  const [activeExercise, setActiveExercise] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [timeLeft, setTimeLeft] = useState(EXERCISES[activeExercise].steps[0].duration);
  const [totalCycles, setTotalCycles] = useState(0);

  const exercise = EXERCISES[activeExercise];
  const step = exercise.steps[currentStep];

  const reset = useCallback(() => {
    setIsPlaying(false);
    setCurrentStep(0);
    setTimeLeft(exercise.steps[0].duration);
    setTotalCycles(0);
  }, [exercise]);

  useEffect(() => {
    reset();
  }, [activeExercise, reset]);

  useEffect(() => {
    let timer: NodeJS.Timeout;
    if (isPlaying) {
      timer = setInterval(() => {
        setTimeLeft((prev) => {
          if (prev <= 1) {
            const nextStep = (currentStep + 1) % exercise.steps.length;
            if (nextStep === 0) setTotalCycles((c) => c + 1);
            setCurrentStep(nextStep);
            return exercise.steps[nextStep].duration;
          }
          return prev - 1;
        });
      }, 1000);
    }
    return () => clearInterval(timer);
  }, [isPlaying, currentStep, exercise]);

  return (
    <Card className="overflow-hidden border-primary/20 bg-gradient-to-br from-background to-primary/5">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Wind className="w-5 h-5 text-primary" />
            <CardTitle>Mindful Breathing</CardTitle>
          </div>
          <div className="flex items-center gap-1">
             <Sparkles className="w-4 h-4 text-amber-500" />
             <span className="text-[10px] font-bold uppercase text-amber-500">All Izz Well</span>
          </div>
        </div>
        <CardDescription>Select a technique and follow the visual guide.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-8">
        <div className="flex gap-2 overflow-x-auto pb-2 scrollbar-hide">
          {EXERCISES.map((ex, i) => (
            <Button
              key={i}
              variant={activeExercise === i ? "default" : "outline"}
              size="sm"
              onClick={() => setActiveExercise(i)}
              className="whitespace-nowrap rounded-full h-8 text-xs"
            >
              {ex.name}
            </Button>
          ))}
        </div>

        <div className="flex flex-col items-center justify-center py-8 relative">
          {/* Animated Circle */}
          <div className="relative w-48 h-48 flex items-center justify-center">
            <AnimatePresence mode="wait">
              <motion.div
                key={step.action}
                initial={{ scale: step.action === "Inhale" ? 0.8 : 1.2, opacity: 0.3 }}
                animate={{ 
                  scale: step.action === "Inhale" ? 1.2 : step.action === "Exhale" ? 0.8 : 1.1,
                  opacity: 1 
                }}
                transition={{ duration: step.duration, ease: "easeInOut" }}
                className={`absolute inset-0 rounded-full border-4 ${
                  step.action === "Inhale" ? "border-primary bg-primary/10" : 
                  step.action === "Hold" ? "border-amber-500 bg-amber-500/10" : 
                  "border-indigo-500 bg-indigo-500/10"
                }`}
              />
            </AnimatePresence>
            
            <div className="z-10 text-center space-y-1">
              <motion.p 
                key={step.action}
                initial={{ y: 5, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                className="text-2xl font-bold tracking-tight"
              >
                {step.action}
              </motion.p>
              <p className="text-4xl font-black tabular-nums">{timeLeft}</p>
            </div>
          </div>

          <div className="mt-8 flex items-center gap-4">
            <Button 
              size="icon" 
              variant="outline" 
              onClick={reset}
              className="rounded-full"
            >
              <RotateCcw className="w-4 h-4" />
            </Button>
            <Button 
              size="lg" 
              onClick={() => setIsPlaying(!isPlaying)}
              className="rounded-full w-32 gap-2"
            >
              {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5 ml-1" />}
              {isPlaying ? "Pause" : "Start"}
            </Button>
            <div className="w-10 text-center">
                <p className="text-[10px] font-bold text-muted-foreground uppercase">Cycles</p>
                <p className="font-bold">{totalCycles}</p>
            </div>
          </div>
        </div>

        <div className="p-4 rounded-xl bg-muted/50 border space-y-2">
            <p className="text-sm font-semibold">{exercise.name}</p>
            <p className="text-xs text-muted-foreground leading-relaxed">{exercise.description}</p>
        </div>
      </CardContent>
    </Card>
  );
}
