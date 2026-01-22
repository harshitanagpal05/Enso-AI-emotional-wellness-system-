"use client";

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Info, HelpCircle } from "lucide-react";

export function AboutEnso() {
  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button variant="ghost" size="icon" className="rounded-full text-muted-foreground hover:text-primary">
          <HelpCircle className="w-5 h-5" />
          <span className="sr-only">About EnsoAI</span>
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <div className="flex items-center gap-4 mb-2">
             <div className="w-12 h-12 rounded-full border-4 border-primary/20 border-t-primary flex items-center justify-center animate-spin-slow">
                <span className="text-primary font-serif text-2xl">円</span>
             </div>
             <div>
                <DialogTitle className="text-2xl font-bold">About EnsoAI</DialogTitle>
                <DialogDescription>The story behind our name and vision</DialogDescription>
             </div>
          </div>
        </DialogHeader>
        <div className="space-y-4 text-sm leading-relaxed text-muted-foreground">
          <p>
            The name <span className="font-bold text-foreground">EnsoAI</span> is derived from the Japanese concept <span className="font-semibold text-primary italic">Enso (円相)</span>, a Zen symbol represented by a hand-drawn circle. Enso symbolizes wholeness, balance, and self-awareness, often drawn in a single, mindful brushstroke to reflect the present state of the mind.
          </p>
          
          <p>
            In the context of this project, <span className="text-foreground">Enso</span> represents the complete emotional state of an individual, acknowledging that human emotions are continuous, dynamic, and interconnected rather than isolated. The circular form signifies how emotions evolve, return, and influence one another over time.
          </p>
          
          <p>
            The suffix <span className="font-bold text-foreground">AI</span> highlights the role of Artificial Intelligence in understanding and interpreting these emotional states. By combining advanced AI techniques with emotion recognition, EnsoAI aims to analyze human emotions holistically and provide personalized insights and recommendations.
          </p>
          
            <div className="bg-primary/5 p-4 rounded-lg border border-primary/10 mt-6">
              <p className="font-semibold text-foreground mb-1">Thus, EnsoAI reflects the project's core objective:</p>
              <p className="italic text-primary">
                to achieve a balanced and comprehensive understanding of human emotions through intelligent systems, promoting emotional awareness, well-being, and personalized user experiences.
              </p>
            </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
