"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Brain, LayoutDashboard, Camera, LogOut, User as UserIcon, HelpCircle, MessageCircle } from "lucide-react";
import { supabase } from "@/lib/supabase";
import { useEffect, useState } from "react";
import { User } from "@supabase/supabase-js";
import { AboutEnso } from "@/components/AboutEnso";

export function Navbar() {
  const pathname = usePathname();
  const [user, setUser] = useState<User | null>(null);

  useEffect(() => {
    supabase.auth.getUser().then(({ data: { user } }) => {
      setUser(user);
    });

    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user ?? null);
    });

    return () => subscription.unsubscribe();
  }, []);

  const handleSignOut = async () => {
    await supabase.auth.signOut();
  };

    return (
        <nav className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 sticky top-0 z-50">
            <div className="w-full px-4 h-16 flex items-center justify-between">
                <Link href="/" className="flex items-center gap-2 font-bold text-xl text-primary shrink-0">
                  <img src="https://slelguoygbfzlpylpxfs.supabase.co/storage/v1/render/image/public/document-uploads/image-1766819359264.png?width=100&height=100&resize=contain" alt="EnsoAI Logo" className="w-10 h-10 object-contain" />
                  <span>EnsoAI</span>
                </Link>
  
              <div className="flex items-center gap-4 md:gap-6 overflow-hidden">
              <Link href="/" className={`text-sm font-medium transition-colors hover:text-primary shrink-0 ${pathname === "/" ? "text-primary" : "text-muted-foreground"}`}>
                <div className="flex items-center gap-1">
                  <Camera className="w-4 h-4" />
                  <span className="hidden sm:inline">Capture</span>
                </div>
              </Link>
              <Link href="/buddy" className={`text-sm font-medium transition-colors hover:text-primary shrink-0 ${pathname === "/buddy" ? "text-primary" : "text-muted-foreground"}`}>
                <div className="flex items-center gap-1">
                  <MessageCircle className="w-4 h-4" />
                  <span className="hidden sm:inline">Enso Buddy</span>
                </div>
              </Link>
              <Link href="/mindfulness" className={`text-sm font-medium transition-colors hover:text-primary shrink-0 ${pathname === "/mindfulness" ? "text-primary" : "text-muted-foreground"}`}>
                <div className="flex items-center gap-1">
                  <Brain className="w-4 h-4" />
                  <span className="hidden sm:inline">Mindfulness</span>
                </div>
              </Link>
              <Link href="/dashboard" className={`text-sm font-medium transition-colors hover:text-primary shrink-0 ${pathname === "/dashboard" ? "text-primary" : "text-muted-foreground"}`}>
                <div className="flex items-center gap-1">
                  <LayoutDashboard className="w-4 h-4" />
                  <span className="hidden sm:inline">Dashboard</span>
                </div>
              </Link>
                <Link href="/profile" className={`text-sm font-medium transition-colors hover:text-primary shrink-0 ${pathname === "/profile" ? "text-primary" : "text-muted-foreground"}`}>
                  <div className="flex items-center gap-1">
                    <UserIcon className="w-4 h-4" />
                    <span className="hidden sm:inline">Profile</span>
                  </div>
                </Link>

                <div className="flex items-center gap-2 border-l pl-4 ml-2 shrink-0">
                  <AboutEnso />
                  {user ? (
                    <Button variant="ghost" size="sm" onClick={handleSignOut} className="gap-2 text-destructive hover:text-destructive hover:bg-destructive/10">
                      <LogOut className="w-4 h-4" />
                      <span className="hidden sm:inline">Sign Out</span>
                    </Button>
                  ) : (
                    <Link href="/auth">
                      <Button size="sm">Sign In</Button>
                    </Link>
                  )}
                </div>
            </div>
          </div>
        </nav>

    );
}
