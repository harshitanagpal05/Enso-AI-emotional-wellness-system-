"use client";

import { useState } from "react";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ThumbsUp, ThumbsDown, Music, Film, Activity, Quote, ExternalLink, Trophy, Sparkles } from "lucide-react";
import { supabase } from "@/lib/supabase";
import { toast } from "sonner";

interface Recommendation {
  id?: string;
  type: string;
  content: string;
  reason: string;
  link?: string;
  feedback_score?: number;
}

export function RecommendationList({ recommendations }: { recommendations: Recommendation[] }) {
  const [feedback, setFeedback] = useState<Record<string, number>>({});

  const handleFeedback = async (index: number, score: number) => {
    const rec = recommendations[index];
    setFeedback(prev => ({ ...prev, [index]: score }));

    try {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) return;

      const { data, error } = await supabase
        .from("recommendations")
        .select("id")
        .eq("content", rec.content)
        .order("created_at", { ascending: false })
        .limit(1)
        .single();

      if (data) {
        await supabase
          .from("recommendations")
          .update({ feedback_score: score })
          .eq("id", data.id);
        
        toast.success(score === 1 ? "Glad you found this helpful!" : "Thanks for the feedback.");
      }
    } catch (err) {
      console.error(err);
    }
  };

    const getIcon = (type: string) => {
      switch (type.toLowerCase()) {
        case "music": return <Music className="w-4 h-4" />;
        case "movie": return <Film className="w-4 h-4" />;
        case "activity": return <Activity className="w-4 h-4" />;
        case "quote": return <Quote className="w-4 h-4" />;
        case "sports": return <Trophy className="w-4 h-4" />;
        case "interactive": return <Sparkles className="w-4 h-4" />;
        default: return <Activity className="w-4 h-4" />;
      }
    };

    const getLinkLabel = (type: string) => {
      switch (type.toLowerCase()) {
        case "music": return "Listen Now";
        case "movie": return "Watch Trailer";
        case "sports": return "View Details";
        default: return "Open Link";
      }
    };


  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Personalized Recommendations</h2>
        <Badge variant="secondary" className="px-3 py-1">AI Curated</Badge>
      </div>

      <div className="grid gap-4">
        {recommendations.map((rec, i) => (
          <Card key={i} className="group hover:border-primary/50 transition-all duration-300">
            <CardContent className="p-5 flex items-start gap-5">
              <div className="p-3 rounded-xl bg-primary/10 text-primary group-hover:bg-primary group-hover:text-primary-foreground transition-colors">
                {getIcon(rec.type)}
              </div>
              
              <div className="flex-1 space-y-2">
                <div className="flex items-center gap-2">
                  <span className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground">{rec.type}</span>
                </div>
                  <div>
                    <h3 className="text-lg font-semibold leading-tight">{rec.content}</h3>
                    <p className="text-sm text-muted-foreground">{rec.reason}</p>
                    {rec.link && (
                      <Button
                        variant="link"
                        size="sm"
                        className="p-0 h-auto text-primary mt-2 gap-1.5 font-bold"
                        asChild
                      >
                        <a href={rec.link} target="_blank" rel="noopener noreferrer">
                          {getLinkLabel(rec.type)}
                          <ExternalLink className="w-3.5 h-3.5" />
                        </a>
                      </Button>
                    )}
                  </div>
                </div>

                <div className="flex flex-col gap-2">
                <Button 
                  size="icon" 
                  variant={feedback[i] === 1 ? "default" : "outline"} 
                  className="h-8 w-8 rounded-full"
                  onClick={() => handleFeedback(i, 1)}
                >
                  <ThumbsUp className="w-3.5 h-3.5" />
                </Button>
                <Button 
                  size="icon" 
                  variant={feedback[i] === -1 ? "destructive" : "outline"} 
                  className="h-8 w-8 rounded-full"
                  onClick={() => handleFeedback(i, -1)}
                >
                  <ThumbsDown className="w-3.5 h-3.5" />
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
