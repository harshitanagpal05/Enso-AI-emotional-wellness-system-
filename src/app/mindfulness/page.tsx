"use client";

import { BreathingExercise } from "@/components/BreathingExercise";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { Sparkles, Leaf, Heart, Brain, Moon, Sun, Wind, Quote } from "lucide-react";

const MINDFULNESS_RESOURCES = [
  {
    title: "Grounding Technique (5-4-3-2-1)",
    description: "A simple technique to bring you back to the present when feeling overwhelmed.",
    icon: Leaf,
    color: "text-emerald-500",
    bg: "bg-emerald-500/10",
    content: [
      "5 things you can see",
      "4 things you can touch",
      "3 things you can hear",
      "2 things you can smell",
      "1 thing you can taste"
    ]
  },
  {
    title: "Morning Routine",
    description: "Start your day with intention and clarity.",
    icon: Sun,
    color: "text-amber-500",
    bg: "bg-amber-500/10",
    content: [
      "Stretch for 5 minutes",
      "Drink a glass of water",
      "Avoid your phone for 30 mins",
      "Set one positive intention",
      "Practice 3 deep breaths"
    ]
  },
  {
    title: "Sleep Hygiene",
    description: "Prepare your mind and body for restful sleep.",
    icon: Moon,
    color: "text-indigo-500",
    bg: "bg-indigo-500/10",
    content: [
      "Dim the lights 1hr before bed",
      "Read a physical book",
      "No caffeine after 3 PM",
      "Keep the room cool",
      "Gentle bedtime stretching"
    ]
  }
];

const QUOTES = [
  "Main apni favorite hoon. - Geet (Jab We Met)",
  "All izz well. - 3 Idiots",
  "The present moment is the only time over which we have dominion. - Thich Nhat Hanh",
  "Breathe. You are exactly where you need to be.",
  "Your calm is your superpower."
];

export default function MindfulnessPage() {
  return (
    <div className="max-w-6xl mx-auto space-y-12 pb-12">
      <div className="text-center space-y-4">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 text-primary text-xs font-bold uppercase tracking-wider">
          <Sparkles className="w-3 h-3" />
          Mindfulness Hub
        </div>
        <h1 className="text-4xl font-bold tracking-tight">Pause. Breathe. Reflect.</h1>
        <p className="text-muted-foreground max-w-2xl mx-auto">
          Take a moment for yourself. Our mindfulness tools are designed to help you regain balance and find peace in the present.
        </p>
      </div>

      <div className="grid lg:grid-cols-5 gap-8">
        <div className="lg:col-span-3">
          <BreathingExercise />
        </div>
        
        <div className="lg:col-span-2 space-y-6">
          <Card className="bg-gradient-to-br from-primary/10 to-transparent border-primary/20">
            <CardHeader className="pb-2">
              <div className="flex items-center gap-2">
                <Quote className="w-4 h-4 text-primary" />
                <CardTitle className="text-sm">Daily Affirmation</CardTitle>
              </div>
            </CardHeader>
            <CardContent>
              <p className="text-xl font-medium italic">
                "{QUOTES[Math.floor(Math.random() * QUOTES.length)]}"
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <div className="flex items-center gap-2">
                <Brain className="w-5 h-5 text-primary" />
                <CardTitle>Why Mindfulness?</CardTitle>
              </div>
              <CardDescription>Scientific benefits of regular practice.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {[
                { label: "Reduces Stress", desc: "Lowers cortisol levels in the brain." },
                { label: "Improves Focus", desc: "Strengthens neural pathways for attention." },
                { label: "Emotional Balance", desc: "Helps regulate responses to triggers." }
              ].map((benefit, i) => (
                <div key={i} className="flex gap-3">
                  <div className="mt-1 p-1 rounded bg-primary/10 text-primary">
                    <Sparkles className="w-3 h-3" />
                  </div>
                  <div>
                    <p className="text-sm font-bold">{benefit.label}</p>
                    <p className="text-xs text-muted-foreground">{benefit.desc}</p>
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        </div>
      </div>

      <div className="grid md:grid-cols-3 gap-6">
        {MINDFULNESS_RESOURCES.map((resource, i) => (
          <Card key={i} className="hover:shadow-lg transition-shadow border-primary/5">
            <CardHeader className="pb-3">
              <div className={`w-10 h-10 rounded-xl ${resource.bg} flex items-center justify-center mb-2`}>
                <resource.icon className={`w-6 h-6 ${resource.color}`} />
              </div>
              <CardTitle className="text-lg">{resource.title}</CardTitle>
              <CardDescription className="text-xs leading-relaxed">
                {resource.description}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2">
                {resource.content.map((item, j) => (
                  <li key={j} className="flex items-center gap-2 text-sm text-muted-foreground">
                    <div className="w-1.5 h-1.5 rounded-full bg-primary/40" />
                    {item}
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
